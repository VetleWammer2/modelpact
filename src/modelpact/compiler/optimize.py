"""Primal-dual optimization of trainable low-rank parameterizations."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from modelpact.compiler.analysis import analyze_candidate_modules, contrastive_gradient_matrix, patchable_linear_weights
from modelpact.compiler.constraints import (
    DifferentiableConstraint,
    DifferentiableObjective,
    MultiplierState,
    augmented_penalty,
    mean_loss,
)
from modelpact.compiler.initialize import gradient_initializer
from modelpact.compiler.result import CompilationResult, CompilationStatus, StepEvidence
from modelpact.util.seeds import seed_everything


class AdditiveLowRankParametrization(nn.Module):
    def __init__(self, left: Tensor, right: Tensor) -> None:
        super().__init__()
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("invalid low-rank factor shapes")
        self.left = nn.Parameter(left.clone())
        self.right = nn.Parameter(right.clone())

    def forward(self, original: Tensor) -> Tensor:
        return original + self.left @ self.right

    @property
    def effective_delta(self) -> Tensor:
        return self.left @ self.right


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    maximum_rank: int = 4
    maximum_modules: int = 4
    steps: int = 200
    learning_rate: float = 1e-2
    multiplier_learning_rate: float = 0.25
    rho: float = 10.0
    gradient_clip: float = 1.0
    complexity_weight: float = 1e-5
    seed: int = 0
    patience: int = 50
    improvement_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if min(self.maximum_rank, self.maximum_modules, self.steps, self.patience) <= 0:
            raise ValueError("rank, module, step, and patience budgets must be positive")
        if min(self.learning_rate, self.multiplier_learning_rate, self.rho, self.gradient_clip) <= 0:
            raise ValueError("optimizer rates and bounds must be positive")


def _install_parameterizations(
    model: nn.Module,
    selected: tuple[str, ...],
    objectives: tuple[DifferentiableObjective, ...],
    guards: tuple[DifferentiableConstraint, ...],
    config: OptimizerConfig,
) -> dict[str, AdditiveLowRankParametrization]:
    modules = patchable_linear_weights(model)
    parameterizations: dict[str, AdditiveLowRankParametrization] = {}
    for offset, name in enumerate(selected):
        module = modules[name]
        rank = min(config.maximum_rank, *module.weight.shape)
        gradient = contrastive_gradient_matrix(model, module, objectives, guards)
        initialized = gradient_initializer(gradient, rank=rank, seed=config.seed + offset)
        parameterization = AdditiveLowRankParametrization(
            initialized.left.to(device=module.weight.device, dtype=module.weight.dtype),
            initialized.right.to(device=module.weight.device, dtype=module.weight.dtype),
        )
        parametrize.register_parametrization(module, "weight", parameterization, unsafe=False)
        parameterizations[name] = parameterization
    return parameterizations


def _snapshot(parameterizations: dict[str, AdditiveLowRankParametrization]) -> dict[str, tuple[Tensor, Tensor]]:
    return {
        name: (item.left.detach().clone(), item.right.detach().clone())
        for name, item in parameterizations.items()
    }


def _restore(parameterizations: dict[str, AdditiveLowRankParametrization], snapshot: dict[str, tuple[Tensor, Tensor]]) -> None:
    with torch.no_grad():
        for name, (left, right) in snapshot.items():
            parameterizations[name].left.copy_(left)
            parameterizations[name].right.copy_(right)


def compile_low_rank_patch(
    base_model: nn.Module,
    objectives: tuple[DifferentiableObjective, ...],
    guards: tuple[DifferentiableConstraint, ...],
    *,
    config: OptimizerConfig = OptimizerConfig(),
) -> CompilationResult:
    """Compile a real additive low-rank patch on a private model copy.

    The caller owns validation and sealed-holdout authorization. This routine
    never inspects holdout data and never mutates ``base_model``.
    """

    if not objectives:
        raise ValueError("at least one target objective is required")
    seed_everything(config.seed)
    model = copy.deepcopy(base_model)
    evidence = analyze_candidate_modules(model, objectives, guards, maximum_modules=config.maximum_modules)
    selected = tuple(item.module_name for item in evidence if item.target_gradient_norm > 0)
    if not selected:
        return CompilationResult(
            status=CompilationStatus.INFEASIBLE_WITHIN_BUDGET,
            deltas={},
            factors={},
            active_modules=(),
            ranks={},
            warnings=["No candidate module had a nonzero target gradient."],
            metadata={"module_evidence": [item.__dict__ for item in evidence]},
        )
    parameterizations = _install_parameterizations(model, selected, objectives, guards, config)
    trainable = [parameter for item in parameterizations.values() for parameter in item.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    multipliers = {
        guard.constraint_id: MultiplierState(learning_rate=config.multiplier_learning_rate)
        for guard in guards
    }
    history: list[StepEvidence] = []
    best_snapshot: dict[str, tuple[Tensor, Tensor]] | None = None
    best_step: int | None = None
    best_loss = float("inf")
    stale_steps = 0
    last_violations: dict[str, float] = {}
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        target_values = {
            objective.objective_id: mean_loss(model, objective.batches, objective.loss) * objective.weight
            for objective in objectives
        }
        target_total = torch.stack(tuple(target_values.values())).sum()
        guard_values = {
            guard.constraint_id: mean_loss(model, guard.batches, guard.measure) - guard.maximum
            for guard in guards
        }
        patch_complexity = torch.stack([item.effective_delta.square().sum() for item in parameterizations.values()]).sum()
        augmented = target_total + config.complexity_weight * patch_complexity
        for identifier, violation in guard_values.items():
            augmented = augmented + augmented_penalty(violation, multipliers[identifier].value, config.rho)
        augmented.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip).item())
        optimizer.step()
        numeric_guards = {identifier: float(value.detach().item()) for identifier, value in guard_values.items()}
        feasible = all(value <= 0 for value in numeric_guards.values())
        numeric_target = {identifier: float(value.detach().item()) for identifier, value in target_values.items()}
        numeric_total = sum(numeric_target.values())
        for identifier, violation in numeric_guards.items():
            multipliers[identifier].update(violation)
        history.append(
            StepEvidence(
                step=step,
                target_losses=numeric_target,
                guard_margins=numeric_guards,
                multipliers={identifier: state.value for identifier, state in multipliers.items()},
                gradient_norm=gradient_norm,
                patch_norm=float(patch_complexity.detach().sqrt().item()),
                feasible=feasible,
            )
        )
        if feasible and numeric_total < best_loss - config.improvement_tolerance:
            best_snapshot = _snapshot(parameterizations)
            best_step = step
            best_loss = numeric_total
            stale_steps = 0
        else:
            stale_steps += 1
        last_violations = numeric_guards
        if best_snapshot is not None and stale_steps >= config.patience:
            break
    if best_snapshot is None:
        snapshot = _snapshot(parameterizations)
        status = CompilationStatus.INFEASIBLE_WITHIN_BUDGET
        warnings = ["No candidate satisfied every declared differentiable constraint within the optimization budget."]
    else:
        _restore(parameterizations, best_snapshot)
        snapshot = best_snapshot
        status = CompilationStatus.FEASIBLE
        warnings = []
    deltas = {name: left @ right for name, (left, right) in snapshot.items()}
    return CompilationResult(
        status=status,
        deltas=deltas,
        factors=snapshot,
        active_modules=tuple(sorted(snapshot)),
        ranks={name: left.shape[1] for name, (left, _) in snapshot.items()},
        evidence=history,
        best_step=best_step,
        best_target_loss=None if best_snapshot is None else best_loss,
        violated_constraints={name: value for name, value in last_violations.items() if value > 0},
        warnings=warnings,
        metadata={
            "optimizer": "AdamW",
            "primal_dual": True,
            "rho": config.rho,
            "seed": config.seed,
            "module_evidence": [
                {
                    "module_name": item.module_name,
                    "parameter_name": item.parameter_name,
                    "shape": list(item.shape),
                    "target_gradient_norm": item.target_gradient_norm,
                    "guard_gradient_norm": item.guard_gradient_norm,
                    "contrastive_score": item.contrastive_score,
                    "parameter_count": item.parameter_count,
                    "estimated_delta_bytes": item.estimated_delta_bytes,
                }
                for item in evidence
            ],
        },
    ).detached_cpu()

