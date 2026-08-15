"""Runtime mounting through PyTorch parametrizations.

The original parameter object is retained by PyTorch's parametrization layer.
Unmounting with ``leave_parametrized=False`` exposes that untouched object again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import Tensor, nn
from torch.nn.utils import parametrize

from modelpact.models.manifest import ModelSignature
from modelpact.models.schema import ModelStateSchema, inspect_state_schema
from modelpact.patch.ast import DeltaProgram
from modelpact.patch.bundle import PatchBundle
from modelpact.patch.validate import validate_base_signature

MOUNT_ATTRIBUTE = "_modelpact_runtime_mount"


class _DeltaParametrization(nn.Module):
    def __init__(
        self,
        program: DeltaProgram,
        target: str,
        tensors: Mapping[str, Tensor],
        shared_factors: Mapping[str, Tensor],
    ) -> None:
        super().__init__()
        self.program = program
        self.target = target
        self._tensor_names = program.referenced_tensors(target)
        self._attribute_by_name: dict[str, str] = {}
        for index, name in enumerate(self._tensor_names):
            attribute = f"factor_{index}"
            self._attribute_by_name[name] = attribute
            value = shared_factors[name]
            if isinstance(value, nn.Parameter):
                self.register_parameter(attribute, value)
            else:
                self.register_buffer(attribute, value, persistent=False)

    def tensor_map(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, attribute) for name, attribute in self._attribute_by_name.items()
        }

    def forward(self, original: Tensor) -> Tensor:
        delta = self.program.materialize(self.target, self.tensor_map())
        if delta.shape != original.shape or delta.dtype != original.dtype:
            raise RuntimeError("mounted delta changed shape or dtype after validation")
        return original + delta


@dataclass(slots=True)
class MountedPatch:
    model: nn.Module
    program: DeltaProgram
    mounted_parameters: tuple[tuple[nn.Module, str], ...]
    _factor_objects: Mapping[str, Tensor]
    _active: bool = field(default=True, init=False)

    @property
    def active(self) -> bool:
        return self._active

    def factor_tensors(self) -> dict[str, Tensor]:
        return dict(self._factor_objects)

    def unmount(self) -> None:
        if not self._active:
            return
        for module, parameter_name in reversed(self.mounted_parameters):
            parametrize.remove_parametrizations(module, parameter_name, leave_parametrized=False)
        if getattr(self.model, MOUNT_ATTRIBUTE, None) is self:
            delattr(self.model, MOUNT_ATTRIBUTE)
        self._active = False

    def __enter__(self) -> MountedPatch:
        if not self._active:
            raise RuntimeError("cannot re-enter an unmounted patch")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.unmount()


def _target_module(model: nn.Module, parameter_path: str) -> tuple[nn.Module, str]:
    module_path, separator, parameter_name = parameter_path.rpartition(".")
    if not separator:
        module, parameter_name = model, parameter_path
    else:
        try:
            module = model.get_submodule(module_path)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"patch target module does not exist: {module_path}") from error
    if parameter_name not in module._parameters or module._parameters[parameter_name] is None:
        raise ValueError(f"patch target is not a direct parameter: {parameter_path}")
    return module, parameter_name


def mount_patch(
    model: nn.Module,
    program: DeltaProgram,
    tensors: Mapping[str, Tensor],
    *,
    state_schema: ModelStateSchema | None = None,
    trainable: bool = False,
) -> MountedPatch:
    """Mount one validated patch without changing any base parameter values."""

    if hasattr(model, MOUNT_ATTRIBUTE):
        raise RuntimeError("a ModelPact patch is already mounted")
    schema = state_schema or inspect_state_schema(model)
    program.validate(tensors, schema)
    shared: dict[str, Tensor] = {}
    for name, tensor in sorted(tensors.items()):
        if tensor.is_floating_point() and trainable:
            shared[name] = (
                tensor
                if isinstance(tensor, nn.Parameter)
                else nn.Parameter(tensor.detach().clone())
            )
        else:
            shared[name] = tensor.detach().clone()
    mounted: list[tuple[nn.Module, str]] = []
    session: MountedPatch | None = None
    try:
        for target in sorted(program.targets):
            module, parameter_name = _target_module(model, target)
            if parametrize.is_parametrized(module, parameter_name):
                raise RuntimeError(f"target already uses a parametrization: {target}")
            expression = _DeltaParametrization(program, target, tensors, shared)
            parametrize.register_parametrization(module, parameter_name, expression)
            mounted.append((module, parameter_name))
        session = MountedPatch(model, program, tuple(mounted), shared)
        setattr(model, MOUNT_ATTRIBUTE, session)
        return session
    except BaseException:
        for module, parameter_name in reversed(mounted):
            parametrize.remove_parametrizations(module, parameter_name, leave_parametrized=False)
        if session is not None and getattr(model, MOUNT_ATTRIBUTE, None) is session:
            delattr(model, MOUNT_ATTRIBUTE)
        raise


def mount_bundle(
    model: nn.Module,
    bundle: PatchBundle,
    actual_signature: ModelSignature | Mapping[str, object],
    *,
    state_schema: ModelStateSchema | None = None,
    trainable: bool = False,
) -> MountedPatch:
    """Validate a bundle's full base identity before mounting its delta."""

    validate_base_signature(bundle.manifest.base_signature, actual_signature)
    return mount_patch(
        model,
        bundle.program,
        bundle.tensors,
        state_schema=state_schema,
        trainable=trainable,
    )
