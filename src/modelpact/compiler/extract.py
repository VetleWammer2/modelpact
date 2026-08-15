"""Selective behavior extraction from a multi-change target teacher."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modelpact.adapters.base import ModelAdapter, ModelBatch
from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
from modelpact.compiler.result import CompilationResult
from modelpact.diff.witnesses import DifferenceWitness


@dataclass(frozen=True, slots=True)
class TeacherBatch:
    batch: ModelBatch
    logits: Tensor


@dataclass(frozen=True, slots=True)
class ExtractionEvidence:
    selected_witness_ids: tuple[str, ...]
    nonselected_witness_ids: tuple[str, ...]
    selected_teacher_kl: float
    nonselected_base_kl: float
    validation_passed: bool
    compiler_result: CompilationResult


def _kl_loss(adapter: ModelAdapter) -> Callable[[nn.Module, TeacherBatch], Tensor]:
    def loss(model: nn.Module, teacher_batch: TeacherBatch) -> Tensor:
        logits = adapter.forward_logits(model, teacher_batch.batch)
        positions = teacher_batch.batch.attention_mask.to(logits.device).sum(dim=1) - 1
        rows = torch.arange(logits.shape[0], device=logits.device)
        logits = logits[rows, positions]
        teacher_logits = teacher_batch.logits.to(device=logits.device)[rows, positions]
        teacher = torch.softmax(teacher_logits.to(dtype=torch.float64), dim=-1).clamp_min(1e-12)
        student_log = torch.log_softmax(logits.to(torch.float64), dim=-1)
        return (teacher * (teacher.log() - student_log)).sum(dim=-1).mean()

    return loss


def _distribution_kl(adapter: ModelAdapter, model: nn.Module, teacher_batch: TeacherBatch) -> float:
    with torch.no_grad():
        logits = adapter.forward_logits(model, teacher_batch.batch)
        positions = teacher_batch.batch.attention_mask.to(logits.device).sum(dim=1) - 1
        rows = torch.arange(logits.shape[0], device=logits.device)
        logits = logits[rows, positions]
        teacher_logits = teacher_batch.logits.to(device=logits.device)[rows, positions]
        teacher = torch.softmax(teacher_logits.to(dtype=torch.float64), dim=-1).clamp_min(1e-12)
        student = torch.softmax(logits.to(torch.float64), dim=-1).clamp_min(1e-12)
        return float((teacher * (teacher.log() - student.log())).sum(dim=-1).mean().item())


def _teacher_batches(
    adapter: ModelAdapter,
    teacher: nn.Module,
    prompts: tuple[str, ...],
) -> tuple[TeacherBatch, ...]:
    batches: list[TeacherBatch] = []
    for prompt in prompts:
        batch = adapter.tokenizer().batch([prompt])
        with torch.no_grad():
            logits = adapter.forward_logits(teacher, batch).detach().cpu()
        batches.append(TeacherBatch(batch, logits))
    return tuple(batches)


def apply_dense_deltas(base_model: nn.Module, deltas: dict[str, Tensor]) -> nn.Module:
    model = copy.deepcopy(base_model)
    modules = dict(model.named_modules())
    with torch.no_grad():
        for module_name, delta in sorted(deltas.items()):
            module = modules.get(module_name)
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    f"compiled delta targets a non-linear or missing module: {module_name}"
                )
            if module.weight.shape != delta.shape:
                raise ValueError(f"compiled delta shape mismatch for {module_name}")
            module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
    return model


def extract_behavior_cluster(
    adapter: ModelAdapter,
    base_model: nn.Module,
    target_model: nn.Module,
    selected: tuple[DifferenceWitness, ...],
    nonselected: tuple[DifferenceWitness, ...],
    *,
    additional_guards: tuple[str, ...] = (),
    optimizer_config: OptimizerConfig | None = None,
    maximum_selected_kl: float = 0.05,
    maximum_nonselected_base_kl: float = 0.02,
) -> ExtractionEvidence:
    """Compile only a selected empirical witness domain.

    The target model teaches selected prompts. The unmodified base teaches all
    nonselected witness prompts and explicit guards, so importing unrelated
    target-model changes is directly penalized and verified.
    """

    if not selected:
        raise ValueError("extraction requires a nonempty selected witness cluster")
    if optimizer_config is None:
        optimizer_config = OptimizerConfig()
    selected_prompts = tuple(witness.minimized_input for witness in selected)
    guard_prompts = tuple(
        dict.fromkeys(
            [
                *(witness.minimized_input for witness in nonselected),
                *additional_guards,
            ]
        )
    )
    target_batches = _teacher_batches(adapter, target_model, selected_prompts)
    base_batches = _teacher_batches(adapter, base_model, guard_prompts) if guard_prompts else ()
    objective = DifferentiableObjective(
        "selected_target_teacher_kl", target_batches, _kl_loss(adapter)
    )
    guards = (
        (
            DifferentiableConstraint(
                "nonselected_base_kl",
                base_batches,
                _kl_loss(adapter),
                maximum=maximum_nonselected_base_kl,
            ),
        )
        if base_batches
        else ()
    )
    result = compile_low_rank_patch(base_model, (objective,), guards, config=optimizer_config)
    if not result.feasible:
        return ExtractionEvidence(
            tuple(item.witness_id for item in selected),
            tuple(item.witness_id for item in nonselected),
            float("inf"),
            float("inf"),
            False,
            result,
        )
    patched = apply_dense_deltas(base_model, result.deltas)
    target_kl = sum(_distribution_kl(adapter, patched, item) for item in target_batches) / len(
        target_batches
    )
    base_kl = (
        sum(_distribution_kl(adapter, patched, item) for item in base_batches) / len(base_batches)
        if base_batches
        else 0.0
    )
    return ExtractionEvidence(
        selected_witness_ids=tuple(item.witness_id for item in selected),
        nonselected_witness_ids=tuple(item.witness_id for item in nonselected),
        selected_teacher_kl=target_kl,
        nonselected_base_kl=base_kl,
        validation_passed=(
            target_kl <= maximum_selected_kl and base_kl <= maximum_nonselected_base_kl
        ),
        compiler_result=result,
    )
