"""Delta Program v1: a small, data-only additive parameter language."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

import torch
from torch import Tensor

from modelpact.models.schema import ModelStateSchema, dtype_name

MAX_PROGRAM_TARGETS = 100_000
MAX_EXPRESSION_DEPTH = 32
MAX_SUM_TERMS = 4096
MAX_TENSOR_NAME_LENGTH = 2048
MAX_DELTA_ELEMENTS = 1 << 34

TensorMap: TypeAlias = Mapping[str, Tensor]
AliasResolver: TypeAlias = Callable[[str], Tensor]


def _validate_tensor_name(name: str) -> None:
    if not isinstance(name, str) or not name or len(name) > MAX_TENSOR_NAME_LENGTH:
        raise ValueError("invalid tensor reference")
    if "\x00" in name or name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
        raise ValueError(f"unsafe tensor reference: {name!r}")


def _tensor(tensors: TensorMap, name: str) -> Tensor:
    _validate_tensor_name(name)
    try:
        value = tensors[name]
    except KeyError as error:
        raise ValueError(f"missing delta tensor: {name}") from error
    if not isinstance(value, Tensor):
        raise TypeError(f"delta value is not a tensor: {name}")
    return value


def _finite_scale(scale: float) -> None:
    if isinstance(scale, bool) or not isinstance(scale, int | float) or not math.isfinite(scale):
        raise ValueError("delta scale must be finite")


class DeltaOp(ABC):
    """One safe additive delta expression."""

    @abstractmethod
    def infer_shape(
        self, tensors: TensorMap, resolve_alias: AliasResolver | None = None
    ) -> tuple[int, ...]:
        """Infer and validate the produced delta shape."""

    @abstractmethod
    def validate(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> None:
        """Validate tensor ranks, dtypes, bounds, and expression semantics."""

    @abstractmethod
    def materialize(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> Tensor:
        """Produce the dense additive delta tensor."""

    def apply(
        self, base: Tensor, tensors: TensorMap, resolve_alias: AliasResolver | None = None
    ) -> Tensor:
        delta = self.materialize(tensors, resolve_alias)
        if tuple(base.shape) != tuple(delta.shape):
            raise ValueError(
                f"base/delta shape mismatch: {tuple(base.shape)} != {tuple(delta.shape)}"
            )
        if base.dtype != delta.dtype:
            raise ValueError(f"base/delta dtype mismatch: {base.dtype} != {delta.dtype}")
        return base + delta

    @abstractmethod
    def estimate_bytes(self, tensors: TensorMap) -> int:
        """Estimate referenced factor bytes, before file-level deduplication."""

    @abstractmethod
    def serialize(self) -> dict[str, object]:
        """Return the canonical JSON-compatible representation."""

    @abstractmethod
    def tensor_names(self) -> tuple[str, ...]:
        """Return referenced SafeTensors keys."""


@dataclass(frozen=True, slots=True)
class LowRankMatrixDelta(DeltaOp):
    left: str
    right: str
    scale: float = 1.0

    def infer_shape(
        self, tensors: TensorMap, resolve_alias: AliasResolver | None = None
    ) -> tuple[int, ...]:
        del resolve_alias
        left, right = _tensor(tensors, self.left), _tensor(tensors, self.right)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("low-rank factors must have shapes [out, rank] and [rank, in]")
        if left.shape[1] <= 0:
            raise ValueError("low-rank delta rank must be positive")
        return left.shape[0], right.shape[1]

    def validate(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> None:
        self.infer_shape(tensors, resolve_alias)
        _finite_scale(self.scale)
        left, right = _tensor(tensors, self.left), _tensor(tensors, self.right)
        if left.dtype != right.dtype or not left.is_floating_point():
            raise ValueError("low-rank factors must share a floating dtype")

    def materialize(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> Tensor:
        self.validate(tensors, resolve_alias)
        return (_tensor(tensors, self.left) @ _tensor(tensors, self.right)) * self.scale

    def estimate_bytes(self, tensors: TensorMap) -> int:
        return sum(
            _tensor(tensors, name).numel() * _tensor(tensors, name).element_size()
            for name in self.tensor_names()
        )

    def serialize(self) -> dict[str, object]:
        return {
            "left": self.left,
            "op": "low_rank_matrix",
            "right": self.right,
            "scale": self.scale,
        }

    def tensor_names(self) -> tuple[str, ...]:
        return self.left, self.right


@dataclass(frozen=True, slots=True)
class SparseMatrixDelta(DeltaOp):
    indices: str
    values: str
    shape: tuple[int, int]
    scale: float = 1.0

    def infer_shape(
        self, tensors: TensorMap, resolve_alias: AliasResolver | None = None
    ) -> tuple[int, ...]:
        del tensors, resolve_alias
        if len(self.shape) != 2 or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in self.shape
        ):
            raise ValueError("sparse matrix shape must contain two positive integers")
        if self.shape[0] * self.shape[1] > MAX_DELTA_ELEMENTS:
            raise ValueError("sparse matrix shape exceeds the delta element limit")
        return self.shape

    def validate(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> None:
        self.infer_shape(tensors, resolve_alias)
        _finite_scale(self.scale)
        indices, values = _tensor(tensors, self.indices), _tensor(tensors, self.values)
        if (
            indices.ndim != 2
            or indices.shape[1] != 2
            or indices.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("sparse indices must be an integer [nnz, 2] tensor")
        if (
            values.ndim != 1
            or values.shape[0] != indices.shape[0]
            or not values.is_floating_point()
        ):
            raise ValueError("sparse values must be a floating [nnz] tensor")
        if indices.numel():
            cpu_indices = indices.detach().cpu().to(torch.int64)
            if bool((cpu_indices < 0).any()):
                raise ValueError("sparse indices may not be negative")
            if bool((cpu_indices[:, 0] >= self.shape[0]).any()) or bool(
                (cpu_indices[:, 1] >= self.shape[1]).any()
            ):
                raise ValueError("sparse index is out of bounds")
            flat = cpu_indices[:, 0] * self.shape[1] + cpu_indices[:, 1]
            if flat.numel() > 1 and not bool(torch.all(flat[1:] > flat[:-1])):
                raise ValueError("sparse indices must be strictly sorted and unique")

    def materialize(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> Tensor:
        self.validate(tensors, resolve_alias)
        indices, values = _tensor(tensors, self.indices), _tensor(tensors, self.values)
        sparse = torch.sparse_coo_tensor(
            indices.to(device=values.device, dtype=torch.int64).transpose(0, 1),
            values * self.scale,
            size=self.shape,
            device=values.device,
            dtype=values.dtype,
            check_invariants=False,
        )
        return sparse.to_dense()

    def estimate_bytes(self, tensors: TensorMap) -> int:
        return sum(
            _tensor(tensors, name).numel() * _tensor(tensors, name).element_size()
            for name in self.tensor_names()
        )

    def serialize(self) -> dict[str, object]:
        return {
            "indices": self.indices,
            "op": "sparse_matrix",
            "scale": self.scale,
            "shape": list(self.shape),
            "values": self.values,
        }

    def tensor_names(self) -> tuple[str, ...]:
        return self.indices, self.values


@dataclass(frozen=True, slots=True)
class VectorDelta(DeltaOp):
    tensor: str
    scale: float = 1.0

    def infer_shape(
        self, tensors: TensorMap, resolve_alias: AliasResolver | None = None
    ) -> tuple[int, ...]:
        del resolve_alias
        value = _tensor(tensors, self.tensor)
        if value.ndim != 1:
            raise ValueError("vector delta tensor must be rank one")
        return (value.shape[0],)

    def validate(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> None:
        self.infer_shape(tensors, resolve_alias)
        _finite_scale(self.scale)
        if not _tensor(tensors, self.tensor).is_floating_point():
            raise ValueError("vector delta must use a floating dtype")

    def materialize(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> Tensor:
        self.validate(tensors, resolve_alias)
        return _tensor(tensors, self.tensor) * self.scale

    def estimate_bytes(self, tensors: TensorMap) -> int:
        value = _tensor(tensors, self.tensor)
        return value.numel() * value.element_size()

    def serialize(self) -> dict[str, object]:
        return {"op": "vector", "scale": self.scale, "tensor": self.tensor}

    def tensor_names(self) -> tuple[str, ...]:
        return (self.tensor,)


@dataclass(frozen=True, slots=True)
class Alias(DeltaOp):
    target: str

    def _resolve(self, resolve_alias: AliasResolver | None) -> Tensor:
        _validate_tensor_name(self.target)
        if resolve_alias is None:
            raise ValueError("alias delta requires a program resolver")
        return resolve_alias(self.target)

    def infer_shape(
        self, tensors: TensorMap, resolve_alias: AliasResolver | None = None
    ) -> tuple[int, ...]:
        del tensors
        return tuple(self._resolve(resolve_alias).shape)

    def validate(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> None:
        del tensors
        self._resolve(resolve_alias)

    def materialize(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> Tensor:
        del tensors
        return self._resolve(resolve_alias)

    def estimate_bytes(self, tensors: TensorMap) -> int:
        del tensors
        return 0

    def serialize(self) -> dict[str, object]:
        return {"op": "alias", "target": self.target}

    def tensor_names(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Sum(DeltaOp):
    terms: tuple[DeltaOp, ...]

    def __post_init__(self) -> None:
        if not self.terms or len(self.terms) > MAX_SUM_TERMS:
            raise ValueError("sum must contain a bounded nonempty term list")

    def infer_shape(
        self, tensors: TensorMap, resolve_alias: AliasResolver | None = None
    ) -> tuple[int, ...]:
        shapes = [term.infer_shape(tensors, resolve_alias) for term in self.terms]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(f"sum term shape mismatch: {shapes}")
        return shapes[0]

    def validate(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> None:
        self.infer_shape(tensors, resolve_alias)
        values = [term.materialize(tensors, resolve_alias) for term in self.terms]
        if any(value.dtype != values[0].dtype for value in values[1:]):
            raise ValueError("sum terms must share a dtype")

    def materialize(self, tensors: TensorMap, resolve_alias: AliasResolver | None = None) -> Tensor:
        values = [term.materialize(tensors, resolve_alias) for term in self.terms]
        shapes = [tuple(value.shape) for value in values]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(f"sum term shape mismatch: {shapes}")
        if any(value.dtype != values[0].dtype for value in values[1:]):
            raise ValueError("sum terms must share a dtype")
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result

    def estimate_bytes(self, tensors: TensorMap) -> int:
        return sum(term.estimate_bytes(tensors) for term in self.terms)

    def serialize(self) -> dict[str, object]:
        return {"op": "sum", "terms": [term.serialize() for term in self.terms]}

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(name for term in self.terms for name in term.tensor_names())


def _exact_fields(value: Mapping[str, object], allowed: set[str]) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown delta operation fields: {sorted(unknown)}")


def parse_delta_op(value: Mapping[str, object], *, depth: int = 0) -> DeltaOp:
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValueError("delta expression exceeds maximum depth")
    operation = value.get("op")
    if operation == "low_rank_matrix":
        _exact_fields(value, {"op", "left", "right", "scale"})
        left, right, scale = value.get("left"), value.get("right"), value.get("scale", 1.0)
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or isinstance(scale, bool)
            or not isinstance(scale, int | float)
        ):
            raise ValueError("malformed low-rank matrix delta")
        return LowRankMatrixDelta(left, right, float(scale))
    if operation == "sparse_matrix":
        _exact_fields(value, {"op", "indices", "values", "shape", "scale"})
        indices, values, shape, scale = (
            value.get("indices"),
            value.get("values"),
            value.get("shape"),
            value.get("scale", 1.0),
        )
        if (
            not isinstance(indices, str)
            or not isinstance(values, str)
            or not isinstance(shape, list)
            or len(shape) != 2
            or not all(not isinstance(item, bool) and isinstance(item, int) for item in shape)
            or isinstance(scale, bool)
            or not isinstance(scale, int | float)
        ):
            raise ValueError("malformed sparse matrix delta")
        return SparseMatrixDelta(indices, values, cast(tuple[int, int], tuple(shape)), float(scale))
    if operation == "vector":
        _exact_fields(value, {"op", "tensor", "scale"})
        tensor, scale = value.get("tensor"), value.get("scale", 1.0)
        if (
            not isinstance(tensor, str)
            or isinstance(scale, bool)
            or not isinstance(scale, int | float)
        ):
            raise ValueError("malformed vector delta")
        return VectorDelta(tensor, float(scale))
    if operation == "alias":
        _exact_fields(value, {"op", "target"})
        target = value.get("target")
        if not isinstance(target, str):
            raise ValueError("malformed alias delta")
        return Alias(target)
    if operation == "sum":
        _exact_fields(value, {"op", "terms"})
        terms = value.get("terms")
        if not isinstance(terms, list) or not all(isinstance(item, Mapping) for item in terms):
            raise ValueError("malformed sum delta")
        return Sum(tuple(parse_delta_op(item, depth=depth + 1) for item in terms))
    raise ValueError(f"unknown delta operation: {operation!r}")


@dataclass(frozen=True, slots=True)
class DeltaProgram:
    targets: Mapping[str, DeltaOp]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError(f"unsupported delta program version: {self.schema_version}")
        if not self.targets or len(self.targets) > MAX_PROGRAM_TARGETS:
            raise ValueError("delta program must have a bounded nonempty target map")
        for name in self.targets:
            _validate_tensor_name(name)

    def _resolver(self, tensors: TensorMap) -> AliasResolver:
        active: set[str] = set()
        cache: dict[str, Tensor] = {}

        def resolve(target: str) -> Tensor:
            if target in cache:
                return cache[target]
            if target in active:
                raise ValueError(f"delta alias cycle at {target}")
            operation = self.targets.get(target)
            if operation is None:
                raise ValueError(f"delta alias refers to unknown target: {target}")
            active.add(target)
            try:
                value = operation.materialize(tensors, resolve)
                cache[target] = value
                return value
            finally:
                active.remove(target)

        return resolve

    def validate(self, tensors: TensorMap, state_schema: ModelStateSchema | None = None) -> None:
        resolver = self._resolver(tensors)
        materialized: dict[str, Tensor] = {}
        for target in sorted(self.targets):
            value = resolver(target)
            materialized[target] = value
            if state_schema is not None:
                specification = state_schema.tensor(target)
                if not specification.patchable:
                    raise ValueError(f"target is not patchable: {target}")
                if tuple(value.shape) != specification.shape:
                    raise ValueError(f"target shape mismatch for {target}")
                if dtype_name(value.dtype) != specification.dtype:
                    raise ValueError(f"target dtype mismatch for {target}")
        if state_schema is not None:
            targeted = set(self.targets)
            for group in state_schema.aliases:
                selected = targeted.intersection(group.members)
                if selected and selected != set(group.members):
                    missing = sorted(set(group.members) - selected)
                    raise ValueError(f"tied parameter patch omits aliases: {missing}")
                if selected:
                    canonical = materialized[group.canonical]
                    for member in group.members[1:]:
                        if not torch.equal(canonical, materialized[member]):
                            raise ValueError(
                                f"inconsistent deltas for tied parameters: {group.members}"
                            )
        referenced = {
            name for operation in self.targets.values() for name in operation.tensor_names()
        }
        unknown = referenced - set(tensors)
        if unknown:
            raise ValueError(f"missing delta tensors: {sorted(unknown)}")
        unused = set(tensors) - referenced
        if unused:
            raise ValueError(f"unreferenced delta tensors are not permitted: {sorted(unused)}")

    def materialize(self, target: str, tensors: TensorMap) -> Tensor:
        if target not in self.targets:
            raise KeyError(target)
        return self._resolver(tensors)(target)

    def referenced_tensors(self, target: str) -> tuple[str, ...]:
        """Return factor keys reachable from one target, following aliases safely."""

        if target not in self.targets:
            raise KeyError(target)
        active: set[str] = set()

        def collect_operation(operation: DeltaOp) -> set[str]:
            if isinstance(operation, Alias):
                return collect_target(operation.target)
            if isinstance(operation, Sum):
                return set().union(*(collect_operation(term) for term in operation.terms))
            return set(operation.tensor_names())

        def collect_target(name: str) -> set[str]:
            if name in active:
                raise ValueError(f"delta alias cycle at {name}")
            operation = self.targets.get(name)
            if operation is None:
                raise ValueError(f"delta alias refers to unknown target: {name}")
            active.add(name)
            try:
                return collect_operation(operation)
            finally:
                active.remove(name)

        return tuple(sorted(collect_target(target)))

    def apply_to_state(
        self,
        state: Mapping[str, Tensor],
        tensors: TensorMap,
        *,
        state_schema: ModelStateSchema | None = None,
    ) -> dict[str, Tensor]:
        self.validate(tensors, state_schema)
        missing = set(self.targets) - set(state)
        if missing:
            raise ValueError(f"checkpoint lacks patch targets: {sorted(missing)}")
        resolver = self._resolver(tensors)
        result: dict[str, Tensor] = {}
        for name in sorted(state):
            base = state[name]
            result[name] = (
                self.targets[name].apply(base, tensors, resolver)
                if name in self.targets
                else base.clone()
            )
        return result

    def estimate_bytes(self, tensors: TensorMap) -> int:
        names = {name for operation in self.targets.values() for name in operation.tensor_names()}
        return sum(
            _tensor(tensors, name).numel() * _tensor(tensors, name).element_size() for name in names
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "targets": {name: self.targets[name].serialize() for name in sorted(self.targets)},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DeltaProgram:
        _exact_fields(value, {"schema_version", "targets"})
        if value.get("schema_version") != 1:
            raise ValueError("unsupported delta program version")
        targets = value.get("targets")
        if not isinstance(targets, Mapping) or not all(
            isinstance(name, str) and isinstance(operation, Mapping)
            for name, operation in targets.items()
        ):
            raise ValueError("malformed delta target map")
        parsed = {
            name: parse_delta_op(operation)
            for name, operation in cast(Mapping[str, Mapping[str, object]], targets).items()
        }
        return cls(dict(sorted(parsed.items())))
