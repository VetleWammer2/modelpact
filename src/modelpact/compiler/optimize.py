"""Primal-dual optimization of trainable low-rank parameterizations."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from modelpact.compiler.analysis import (
    analyze_candidate_modules,
    contrastive_gradient_matrix,
    patchable_linear_weights,
)
from modelpact.compiler.constraints import (
    DifferentiableConstraint,
    DifferentiableObjective,
    MultiplierState,
    augmented_penalty,
    mean_loss,
)
from modelpact.compiler.initialize import gradient_initializer, target_delta_initializer
from modelpact.compiler.result import CompilationResult, CompilationStatus, StepEvidence
from modelpact.models.aliases import discover_parameter_aliases
from modelpact.util.seeds import seed_everything


class AdditiveLowRankParametrization(nn.Module):
    def __init__(
        self,
        left: Tensor,
        right: Tensor,
        *,
        share_parameters: bool = False,
    ) -> None:
        super().__init__()
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("invalid low-rank factor shapes")
        if share_parameters:
            if not isinstance(left, nn.Parameter) or not isinstance(right, nn.Parameter):
                raise TypeError("shared factors must already be Parameters")
            self.left = left
            self.right = right
        else:
            self.left = nn.Parameter(left.clone())
            self.right = nn.Parameter(right.clone())

    def forward(self, original: Tensor) -> Tensor:
        return original + self.left @ self.right

    @property
    def effective_delta(self) -> Tensor:
        return self.left @ self.right


def _direct_parameter(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    module_path, separator, parameter_name = path.rpartition(".")
    module = model.get_submodule(module_path) if separator else model
    parameter = module._parameters.get(parameter_name)
    if parameter is None:
        raise ValueError(f"alias target is not a direct parameter: {path}")
    return module, parameter_name


def _resolve_alias_targets(
    model: nn.Module,
    selected: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Deduplicate physical parameters and expand each to all declared aliases."""

    group_by_member = {
        member: group for group in discover_parameter_aliases(model) for member in group.members
    }
    resolved: dict[str, tuple[str, ...]] = {}
    selected_physical: set[str] = set()
    for module_name in selected:
        parameter_name = f"{module_name}.weight"
        group = group_by_member.get(parameter_name)
        physical_name = group.canonical if group is not None else parameter_name
        if physical_name in selected_physical:
            continue
        selected_physical.add(physical_name)
        resolved[module_name] = group.members if group is not None else (parameter_name,)
    return resolved


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
        if (
            min(self.learning_rate, self.multiplier_learning_rate, self.rho, self.gradient_clip)
            <= 0
        ):
            raise ValueError("optimizer rates and bounds must be positive")


DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()


def _install_parameterizations(
    model: nn.Module,
    selected: tuple[str, ...],
    objectives: tuple[DifferentiableObjective, ...],
    guards: tuple[DifferentiableConstraint, ...],
    config: OptimizerConfig,
    alias_targets: dict[str, tuple[str, ...]],
    target_state: Mapping[str, Tensor] | None,
) -> tuple[dict[str, AdditiveLowRankParametrization], dict[str, dict[str, object]]]:
    modules = patchable_linear_weights(model)
    parameterizations: dict[str, AdditiveLowRankParametrization] = {}
    initialization_evidence: dict[str, dict[str, object]] = {}
    for offset, name in enumerate(selected):
        module = modules[name]
        rank = min(config.maximum_rank, *module.weight.shape)
        if target_state is None:
            gradient = contrastive_gradient_matrix(model, module, objectives, guards)
            initialized = gradient_initializer(gradient, rank=rank, seed=config.seed + offset)
        else:
            parameter_name = f"{name}.weight"
            target_weight = target_state.get(parameter_name)
            if target_weight is None:
                raise ValueError(
                    f"target-delta initializer is missing target parameter: {parameter_name}"
                )
            initialized = target_delta_initializer(
                module.weight.detach(),
                target_weight.detach().to(device=module.weight.device, dtype=module.weight.dtype),
                rank=rank,
                seed=config.seed + offset,
            )
        initialization_evidence[name] = {
            "rank": rank,
            "source": initialized.source,
            "source_norm": initialized.source_norm,
        }
        parameterization = AdditiveLowRankParametrization(
            initialized.left.to(device=module.weight.device, dtype=module.weight.dtype),
            initialized.right.to(device=module.weight.device, dtype=module.weight.dtype),
        )
        targets = alias_targets[name]
        for target_index, target in enumerate(targets):
            target_module, target_parameter = _direct_parameter(model, target)
            expression = (
                parameterization
                if target_index == 0
                else AdditiveLowRankParametrization(
                    parameterization.left,
                    parameterization.right,
                    share_parameters=True,
                )
            )
            parametrize.register_parametrization(
                target_module,
                target_parameter,
                expression,
                unsafe=False,
            )
        parameterizations[name] = parameterization
    return parameterizations, initialization_evidence


def _snapshot(
    parameterizations: dict[str, AdditiveLowRankParametrization],
) -> dict[str, tuple[Tensor, Tensor]]:
    return {
        name: (item.left.detach().clone(), item.right.detach().clone())
        for name, item in parameterizations.items()
    }


def _restore(
    parameterizations: dict[str, AdditiveLowRankParametrization],
    snapshot: dict[str, tuple[Tensor, Tensor]],
) -> None:
    with torch.no_grad():
        for name, (left, right) in snapshot.items():
            parameterizations[name].left.copy_(left)
            parameterizations[name].right.copy_(right)


def _evaluate_candidate(
    model: nn.Module,
    objectives: tuple[DifferentiableObjective, ...],
    guards: tuple[DifferentiableConstraint, ...],
    parameterizations: dict[str, AdditiveLowRankParametrization],
) -> tuple[dict[str, float], dict[str, float], float]:
    """Evaluate the exact currently mounted factors without updating state."""

    with torch.no_grad():
        target_values = {
            objective.objective_id: float(
                (mean_loss(model, objective.batches, objective.loss) * objective.weight).item()
            )
            for objective in objectives
        }
        guard_values = {
            guard.constraint_id: float(
                (mean_loss(model, guard.batches, guard.measure) - guard.maximum).item()
            )
            for guard in guards
        }
        complexity = float(
            torch.stack(
                [item.effective_delta.square().sum() for item in parameterizations.values()]
            )
            .sum()
            .item()
        )
    return target_values, guard_values, complexity


def compile_low_rank_patch(
    base_model: nn.Module,
    objectives: tuple[DifferentiableObjective, ...],
    guards: tuple[DifferentiableConstraint, ...],
    *,
    config: OptimizerConfig = DEFAULT_OPTIMIZER_CONFIG,
    target_state: Mapping[str, Tensor] | None = None,
) -> CompilationResult:
    """Compile a real additive low-rank patch on a private model copy.

    The caller owns validation and sealed-holdout authorization. This routine
    never inspects holdout data and never mutates ``base_model``.
    """

    if not objectives:
        raise ValueError("at least one target objective is required")
    seed_everything(config.seed)
    model = copy.deepcopy(base_model)
    evidence = analyze_candidate_modules(
        model, objectives, guards, maximum_modules=config.maximum_modules
    )
    ranked_selected = tuple(item.module_name for item in evidence if item.target_gradient_norm > 0)
    if not ranked_selected:
        return CompilationResult(
            status=CompilationStatus.INFEASIBLE_WITHIN_BUDGET,
            deltas={},
            factors={},
            active_modules=(),
            ranks={},
            warnings=["No candidate module had a nonzero target gradient."],
            metadata={
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
                ]
            },
        )
    alias_targets = _resolve_alias_targets(model, ranked_selected)
    selected = tuple(alias_targets)
    parameterizations, initialization_evidence = _install_parameterizations(
        model,
        selected,
        objectives,
        guards,
        config,
        alias_targets,
        target_state,
    )
    trainable = [
        parameter for item in parameterizations.values() for parameter in item.parameters()
    ]
    if len({id(parameter) for parameter in trainable}) != len(trainable):
        raise RuntimeError("compiler optimizer received duplicate factor parameters")
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
    initial_targets, initial_guards, _ = _evaluate_candidate(
        model, objectives, guards, parameterizations
    )
    last_violations = initial_guards
    if all(value <= 0 for value in initial_guards.values()):
        best_snapshot = _snapshot(parameterizations)
        best_step = -1
        best_loss = sum(initial_targets.values())
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        target_values = {
            objective.objective_id: mean_loss(model, objective.batches, objective.loss)
            * objective.weight
            for objective in objectives
        }
        target_total = torch.stack(tuple(target_values.values())).sum()
        guard_values = {
            guard.constraint_id: mean_loss(model, guard.batches, guard.measure) - guard.maximum
            for guard in guards
        }
        patch_complexity = torch.stack(
            [item.effective_delta.square().sum() for item in parameterizations.values()]
        ).sum()
        augmented = target_total + config.complexity_weight * patch_complexity
        for identifier, guard_violation in guard_values.items():
            augmented = augmented + augmented_penalty(
                guard_violation, multipliers[identifier].value, config.rho
            )
        torch.autograd.backward(augmented)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip).item()
        )
        optimizer.step()
        numeric_target, numeric_guards, numeric_complexity = _evaluate_candidate(
            model, objectives, guards, parameterizations
        )
        feasible = all(value <= 0 for value in numeric_guards.values())
        numeric_total = sum(numeric_target.values())
        for identifier, numeric_violation in numeric_guards.items():
            multipliers[identifier].update(numeric_violation)
        history.append(
            StepEvidence(
                step=step,
                target_losses=numeric_target,
                guard_margins=numeric_guards,
                multipliers={identifier: state.value for identifier, state in multipliers.items()},
                gradient_norm=gradient_norm,
                patch_norm=numeric_complexity**0.5,
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
        warnings = [
            "No candidate satisfied every declared differentiable constraint within the "
            "optimization budget."
        ]
        final_violations = last_violations
    else:
        _restore(parameterizations, best_snapshot)
        snapshot = best_snapshot
        restored_target_values, final_violations, _ = _evaluate_candidate(
            model, objectives, guards, parameterizations
        )
        best_loss = sum(restored_target_values.values())
        if all(value <= 0 for value in final_violations.values()):
            status = CompilationStatus.FEASIBLE
            warnings = []
        else:
            status = CompilationStatus.INFEASIBLE_WITHIN_BUDGET
            warnings = [
                "The selected candidate failed exact guard re-evaluation after restoration."
            ]
    deltas = {name: left @ right for name, (left, right) in snapshot.items()}
    return CompilationResult(
        status=status,
        deltas=deltas,
        factors=snapshot,
        active_modules=tuple(sorted(snapshot)),
        ranks={name: left.shape[1] for name, (left, _) in snapshot.items()},
        evidence=history,
        best_step=best_step,
        best_target_loss=best_loss if status is CompilationStatus.FEASIBLE else None,
        violated_constraints={name: value for name, value in final_violations.items() if value > 0},
        warnings=warnings,
        metadata={
            "optimizer": "AdamW",
            "primal_dual": True,
            "rho": config.rho,
            "seed": config.seed,
            "initial_candidate_evaluated": True,
            "initialization": initialization_evidence,
            "optimized_aliases": {
                name: list(alias_targets[name]) for name in sorted(alias_targets)
            },
            "trainable_factor_parameters": sum(parameter.numel() for parameter in trainable),
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
