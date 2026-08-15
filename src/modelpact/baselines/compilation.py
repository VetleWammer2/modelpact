"""Patch-compilation baselines with explicit optimization budgets."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modelpact.compiler.analysis import (
    ModuleEvidence,
    analyze_candidate_modules,
    patchable_linear_weights,
)
from modelpact.compiler.constraints import DifferentiableObjective, mean_loss
from modelpact.compiler.gradient_basis import low_rank_factors


@dataclass(frozen=True, slots=True)
class FullFineTuneResult:
    model: nn.Module
    delta: dict[str, Tensor]
    losses: tuple[float, ...]
    steps: int


def full_fine_tune(
    base_model: nn.Module,
    objectives: tuple[DifferentiableObjective, ...],
    *,
    steps: int,
    learning_rate: float,
    seed: int = 0,
) -> FullFineTuneResult:
    if not objectives or steps <= 0 or learning_rate <= 0:
        raise ValueError("full fine-tuning requires objectives and positive budgets")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = copy.deepcopy(base_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        losses: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(
                [
                    mean_loss(model, objective.batches, objective.loss) * objective.weight
                    for objective in objectives
                ]
            ).sum()
            loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub gap.
            optimizer.step()
            losses.append(float(loss.detach().item()))
    base_state = base_model.state_dict()
    delta = {
        name: value.detach().cpu() - base_state[name].detach().cpu()
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    return FullFineTuneResult(model, delta, tuple(losses), steps)


def random_module_ranking(model: nn.Module, *, seed: int = 0) -> tuple[str, ...]:
    names = sorted(patchable_linear_weights(model))
    random.Random(seed).shuffle(names)  # noqa: S311 - deterministic research baseline
    return tuple(names)


def gradient_saliency_ranking(
    model: nn.Module,
    objectives: tuple[DifferentiableObjective, ...],
) -> tuple[ModuleEvidence, ...]:
    return analyze_candidate_modules(model, objectives, ())


def truncated_target_delta(
    base_model: nn.Module,
    target_model: nn.Module,
    *,
    rank: int,
) -> dict[str, Tensor]:
    base_modules = patchable_linear_weights(base_model)
    target_modules = patchable_linear_weights(target_model)
    if set(base_modules) != set(target_modules):
        raise ValueError("target-delta SVD requires identical linear module schemas")
    output: dict[str, Tensor] = {}
    for name in sorted(base_modules):
        base = base_modules[name].weight.detach()
        target = target_modules[name].weight.detach()
        if base.shape != target.shape:
            raise ValueError(f"module shape differs: {name}")
        actual_rank = min(rank, *base.shape)
        left, right = low_rank_factors(target - base, rank=actual_rank)
        output[name] = (left @ right).cpu()
    return output
