"""Gradient and target-delta low-rank initializers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from modelpact.compiler.gradient_basis import low_rank_factors


@dataclass(frozen=True, slots=True)
class InitializedFactors:
    left: Tensor
    right: Tensor
    source: str
    source_norm: float


def gradient_initializer(
    gradient: Tensor,
    *,
    rank: int,
    scale: float = 1e-3,
    seed: int = 0,
) -> InitializedFactors:
    left, right = low_rank_factors(-gradient, rank=rank, seed=seed)
    root_scale = abs(scale) ** 0.5
    sign = -1.0 if scale < 0 else 1.0
    return InitializedFactors(
        left * root_scale,
        right * root_scale * sign,
        "contrastive_gradient",
        float(gradient.norm().item()),
    )


def target_delta_initializer(
    base: Tensor,
    target: Tensor,
    *,
    rank: int,
    scale: float = 1.0,
    seed: int = 0,
) -> InitializedFactors:
    if base.shape != target.shape or base.ndim != 2:
        raise ValueError("matching matrix-shaped base and target weights are required")
    delta = target.detach() - base.detach()
    left, right = low_rank_factors(delta, rank=rank, seed=seed)
    root_scale = abs(scale) ** 0.5
    sign = -1.0 if scale < 0 else 1.0
    return InitializedFactors(
        left * root_scale,
        right * root_scale * sign,
        "target_delta_signal",
        float(delta.norm().item()),
    )
