"""Evidence-based candidate module analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modelpact.compiler.constraints import (
    DifferentiableConstraint,
    DifferentiableObjective,
    mean_loss,
)


@dataclass(frozen=True, slots=True)
class ModuleEvidence:
    module_name: str
    parameter_name: str
    shape: tuple[int, ...]
    target_gradient_norm: float
    guard_gradient_norm: float
    contrastive_score: float
    parameter_count: int
    estimated_delta_bytes: int


def patchable_linear_weights(model: nn.Module) -> dict[str, nn.Linear]:
    return {
        name: module
        for name, module in sorted(model.named_modules())
        if name and isinstance(module, nn.Linear) and module.weight.ndim == 2
    }


def _aggregate_gradient(
    model: nn.Module,
    parameter: Tensor,
    losses: Iterable[Tensor],
) -> Tensor:
    aggregate = torch.zeros_like(parameter, memory_format=torch.preserve_format)
    count = 0
    for loss in losses:
        gradient = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True)[0]
        if gradient is not None:
            aggregate += gradient.detach()
        count += 1
    return aggregate / max(1, count)


def analyze_candidate_modules(
    model: nn.Module,
    objectives: tuple[DifferentiableObjective, ...],
    guards: tuple[DifferentiableConstraint, ...],
    *,
    epsilon: float = 1e-12,
    maximum_modules: int | None = None,
) -> tuple[ModuleEvidence, ...]:
    """Rank linear modules by target-sensitive, guard-insensitive gradients."""

    modules = patchable_linear_weights(model)
    if not modules:
        raise ValueError("model exposes no patchable linear weights")
    target_losses = [
        mean_loss(model, objective.batches, objective.loss) * objective.weight
        for objective in objectives
    ]
    guard_losses = [mean_loss(model, guard.batches, guard.measure) for guard in guards]
    evidence: list[ModuleEvidence] = []
    for name, module in modules.items():
        target_gradient = _aggregate_gradient(model, module.weight, target_losses)
        guard_gradient = (
            _aggregate_gradient(model, module.weight, guard_losses)
            if guard_losses
            else torch.zeros_like(module.weight)
        )
        target_norm = float(torch.linalg.vector_norm(target_gradient.to(torch.float64)).item())
        guard_norm = float(torch.linalg.vector_norm(guard_gradient.to(torch.float64)).item())
        evidence.append(
            ModuleEvidence(
                module_name=name,
                parameter_name=f"{name}.weight",
                shape=tuple(module.weight.shape),
                target_gradient_norm=target_norm,
                guard_gradient_norm=guard_norm,
                contrastive_score=target_norm / (epsilon + guard_norm),
                parameter_count=module.weight.numel(),
                estimated_delta_bytes=module.weight.numel() * module.weight.element_size(),
            )
        )
    ordered = sorted(
        evidence,
        key=lambda item: (
            -item.contrastive_score,
            -item.target_gradient_norm,
            item.module_name,
        ),
    )
    if maximum_modules is not None:
        if maximum_modules <= 0:
            raise ValueError("maximum_modules must be positive")
        ordered = ordered[:maximum_modules]
    return tuple(ordered)


def contrastive_gradient_matrix(
    model: nn.Module,
    module: nn.Linear,
    objectives: tuple[DifferentiableObjective, ...],
    guards: tuple[DifferentiableConstraint, ...],
    *,
    guard_projection: float = 1.0,
) -> Tensor:
    target_losses = [
        mean_loss(model, objective.batches, objective.loss) * objective.weight
        for objective in objectives
    ]
    target = _aggregate_gradient(model, module.weight, target_losses)
    if not guards:
        return target
    guard_losses = [mean_loss(model, guard.batches, guard.measure) for guard in guards]
    guard = _aggregate_gradient(model, module.weight, guard_losses)
    denominator = guard.square().sum().clamp_min(1e-12)
    projection = (target * guard).sum() / denominator
    return target - guard_projection * projection * guard
