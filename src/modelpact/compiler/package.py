"""Translate compiler factors into the safe Delta Program and bundle schema."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from modelpact.compiler.result import CompilationResult
from modelpact.models.schema import ModelStateSchema
from modelpact.patch.ast import Alias, DeltaOp, DeltaProgram, LowRankMatrixDelta


def compilation_delta_program(
    result: CompilationResult,
    state_schema: ModelStateSchema,
) -> tuple[DeltaProgram, dict[str, Tensor]]:
    if not result.feasible or not result.factors:
        raise ValueError("only a feasible nonempty compilation result can be packaged")
    targets: dict[str, DeltaOp] = {}
    tensors: dict[str, Tensor] = {}
    for module_name, (left, right) in sorted(result.factors.items()):
        parameter_name = f"{module_name}.weight"
        state_schema.tensor(parameter_name)
        left_key = f"factors.{module_name}.left"
        right_key = f"factors.{module_name}.right"
        tensors[left_key] = left.detach().cpu().contiguous()
        tensors[right_key] = right.detach().cpu().contiguous()
        targets[parameter_name] = LowRankMatrixDelta(left_key, right_key)
    for group in state_schema.aliases:
        selected = set(group.members).intersection(targets)
        if not selected:
            continue
        if len(selected) > 1:
            raise ValueError(f"compiler independently targeted tied aliases: {group.members}")
        selected_name = next(iter(selected))
        operation = targets.pop(selected_name)
        targets[group.canonical] = operation
        for member in group.members:
            if member != group.canonical:
                targets[member] = Alias(group.canonical)
    program = DeltaProgram(dict(sorted(targets.items())))
    program.validate(tensors, state_schema)
    return program, tensors


def compile_evidence(result: CompilationResult) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": result.status.value,
        "active_modules": list(result.active_modules),
        "ranks": dict(sorted(result.ranks.items())),
        "best_step": result.best_step,
        "best_target_loss": result.best_target_loss,
        "violated_constraints": dict(sorted(result.violated_constraints.items())),
        "warnings": list(result.warnings),
        "metadata": result.metadata,
        "steps": [
            {
                "step": item.step,
                "target_losses": dict(sorted(item.target_losses.items())),
                "guard_margins": dict(sorted(item.guard_margins.items())),
                "multipliers": dict(sorted(item.multipliers.items())),
                "gradient_norm": item.gradient_norm,
                "patch_norm": item.patch_norm,
                "feasible": item.feasible,
            }
            for item in result.evidence
        ],
    }


def dense_delta_bytes(deltas: Mapping[str, Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in deltas.values())
