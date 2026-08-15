"""Canonical, architecture-independent model state schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from torch import nn

from modelpact.models.aliases import AliasGroup, discover_parameter_aliases
from modelpact.util.hashing import hash_canonical

MAX_TENSOR_DIMENSIONS = 8
MAX_TENSOR_ELEMENTS = 1 << 40


def dtype_name(dtype: object) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    patchable: bool
    kind: str

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 1024:
            raise ValueError("invalid tensor name")
        if len(self.shape) > MAX_TENSOR_DIMENSIONS:
            raise ValueError(f"invalid shape for {self.name}")
        elements = 1
        for dimension in self.shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"invalid dimension for {self.name}")
            elements *= dimension
            if elements > MAX_TENSOR_ELEMENTS:
                raise ValueError(f"tensor too large in schema: {self.name}")
        if not self.dtype or not self.kind:
            raise ValueError("dtype and tensor kind are required")

    def to_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "kind": self.kind,
            "name": self.name,
            "patchable": self.patchable,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TensorSpec:
        name = value.get("name")
        shape = value.get("shape")
        dtype = value.get("dtype")
        patchable = value.get("patchable")
        kind = value.get("kind")
        if (
            not isinstance(name, str)
            or not isinstance(shape, list)
            or not all(isinstance(item, int) for item in shape)
            or not isinstance(dtype, str)
            or not isinstance(patchable, bool)
            or not isinstance(kind, str)
        ):
            raise ValueError("malformed tensor specification")
        return cls(name, tuple(cast(list[int], shape)), dtype, patchable, kind)


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    path: str
    module_type: str
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.path) > 1024 or not self.module_type:
            raise ValueError("invalid module specification")
        if tuple(sorted(set(self.parameter_names))) != self.parameter_names:
            raise ValueError("module parameter names must be unique and sorted")

    def to_dict(self) -> dict[str, object]:
        return {
            "module_type": self.module_type,
            "parameter_names": list(self.parameter_names),
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModuleSpec:
        path = value.get("path")
        module_type = value.get("module_type")
        names = value.get("parameter_names")
        if (
            not isinstance(path, str)
            or not isinstance(module_type, str)
            or not isinstance(names, list)
            or not all(isinstance(item, str) for item in names)
        ):
            raise ValueError("malformed module specification")
        return cls(path, module_type, tuple(cast(list[str], names)))


@dataclass(frozen=True, slots=True)
class ModelStateSchema:
    schema_version: int
    tensors: tuple[TensorSpec, ...]
    modules: tuple[ModuleSpec, ...]
    aliases: tuple[AliasGroup, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError(f"unsupported model state schema version: {self.schema_version}")
        names = tuple(item.name for item in self.tensors)
        if tuple(sorted(set(names))) != names:
            raise ValueError("tensor specifications must be uniquely sorted")
        module_paths = tuple(item.path for item in self.modules)
        if tuple(sorted(set(module_paths))) != module_paths:
            raise ValueError("module specifications must be uniquely sorted")
        known = set(names)
        for group in self.aliases:
            if not set(group.members) <= known:
                raise ValueError("alias group refers to an unknown tensor")

    @property
    def schema_hash(self) -> str:
        return hash_canonical(self.to_dict())

    def tensor(self, name: str) -> TensorSpec:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "aliases": [group.to_dict() for group in self.aliases],
            "modules": [module.to_dict() for module in self.modules],
            "schema_version": self.schema_version,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelStateSchema:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported model state schema")
        tensors = value.get("tensors")
        modules = value.get("modules")
        aliases = value.get("aliases", [])
        if (
            not isinstance(tensors, list)
            or not isinstance(modules, list)
            or not isinstance(aliases, list)
        ):
            raise ValueError("malformed model state schema")
        if not all(isinstance(item, Mapping) for item in tensors + modules + aliases):
            raise ValueError("malformed model state schema entry")
        return cls(
            schema_version=1,
            tensors=tuple(TensorSpec.from_dict(item) for item in tensors),
            modules=tuple(ModuleSpec.from_dict(item) for item in modules),
            aliases=tuple(AliasGroup.from_dict(item) for item in aliases),
        )


def _parameter_kind(module: nn.Module, local_name: str) -> tuple[bool, str]:
    if isinstance(module, nn.Linear):
        return True, "linear_weight" if local_name == "weight" else "linear_bias"
    if isinstance(module, nn.Embedding) and local_name == "weight":
        return True, "token_embedding"
    if isinstance(module, nn.LayerNorm) and local_name in {"weight", "bias"}:
        return True, "norm_scale" if local_name == "weight" else "norm_bias"
    # TinyRMSNorm and third-party RMSNorm implementations intentionally use a
    # structural check; adapters remain responsible for narrowing this if needed.
    if "rmsnorm" in type(module).__name__.lower() and local_name == "weight":
        return True, "norm_scale"
    return False, "persistent_parameter"


def inspect_state_schema(model: nn.Module) -> ModelStateSchema:
    """Inspect persistent parameters without relying on architecture names."""

    tensor_specs: list[TensorSpec] = []
    module_specs: list[ModuleSpec] = []
    for module_path, module in model.named_modules():
        local_parameters = []
        for local_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_path}.{local_name}" if module_path else local_name
            patchable, kind = _parameter_kind(module, local_name)
            tensor_specs.append(
                TensorSpec(
                    name=full_name,
                    shape=tuple(parameter.shape),
                    dtype=dtype_name(parameter.dtype),
                    patchable=patchable,
                    kind=kind,
                )
            )
            local_parameters.append(full_name)
        for local_name, buffer in module.named_buffers(recurse=False):
            full_name = f"{module_path}.{local_name}" if module_path else local_name
            tensor_specs.append(
                TensorSpec(
                    name=full_name,
                    shape=tuple(buffer.shape),
                    dtype=dtype_name(buffer.dtype),
                    patchable=False,
                    kind="persistent_buffer",
                )
            )
        if local_parameters:
            module_specs.append(
                ModuleSpec(
                    path=module_path,
                    module_type=f"{type(module).__module__}.{type(module).__qualname__}",
                    parameter_names=tuple(sorted(local_parameters)),
                )
            )
    return ModelStateSchema(
        schema_version=1,
        tensors=tuple(sorted(tensor_specs, key=lambda item: item.name)),
        modules=tuple(sorted(module_specs, key=lambda item: item.path)),
        aliases=discover_parameter_aliases(model),
    )
