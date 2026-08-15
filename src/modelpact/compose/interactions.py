"""Structural and semantic patch-interaction diagnostics.

These diagnostics are evidence for search and reporting.  They are never used as
a substitute for executing the union of behavior contracts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch


@dataclass(frozen=True, slots=True)
class SetOverlap:
    left_count: int
    right_count: int
    intersection: tuple[str, ...]
    union_count: int
    jaccard: float


@dataclass(frozen=True, slots=True)
class SparseIndexOverlap:
    module: str
    left_count: int
    right_count: int
    intersection_count: int
    jaccard: float


@dataclass(frozen=True, slots=True)
class PrincipalAngles:
    radians: tuple[float, ...]
    cosines: tuple[float, ...]
    left_rank: int
    right_rank: int

    @property
    def smallest_radians(self) -> float | None:
        return min(self.radians) if self.radians else None


@dataclass(frozen=True, slots=True)
class LowRankSubspaceDiagnostics:
    column_space: PrincipalAngles
    row_space: PrincipalAngles


def module_overlap(left: Sequence[str], right: Sequence[str]) -> SetOverlap:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    intersection = tuple(sorted(left_set & right_set))
    return SetOverlap(
        left_count=len(left_set),
        right_count=len(right_set),
        intersection=intersection,
        union_count=len(union),
        jaccard=len(intersection) / len(union) if union else 1.0,
    )


def sparse_index_overlap(
    left: Mapping[str, Sequence[int]], right: Mapping[str, Sequence[int]]
) -> tuple[SparseIndexOverlap, ...]:
    reports: list[SparseIndexOverlap] = []
    for module in sorted(set(left) | set(right)):
        left_set = set(left.get(module, ()))
        right_set = set(right.get(module, ()))
        union = left_set | right_set
        intersection = left_set & right_set
        reports.append(
            SparseIndexOverlap(
                module=module,
                left_count=len(left_set),
                right_count=len(right_set),
                intersection_count=len(intersection),
                jaccard=len(intersection) / len(union) if union else 1.0,
            )
        )
    return tuple(reports)


def _orthonormal_columns(matrix: torch.Tensor, *, tolerance: float) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("a subspace basis must be a rank-two tensor")
    work = matrix.detach().to(device="cpu", dtype=torch.float64)
    if work.numel() == 0 or min(work.shape) == 0:
        return torch.empty((work.shape[0], 0), dtype=torch.float64)
    left, singular_values, _ = torch.linalg.svd(work, full_matrices=False)
    if singular_values.numel() == 0:
        return torch.empty((work.shape[0], 0), dtype=torch.float64)
    cutoff = tolerance * max(float(singular_values[0]), 1.0)
    rank = int(torch.count_nonzero(singular_values > cutoff).item())
    return cast(torch.Tensor, left[:, :rank])


def principal_angles(
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> PrincipalAngles:
    """Compute principal angles for bases stored as ambient-by-rank matrices."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if left_basis.ndim != 2 or right_basis.ndim != 2:
        raise ValueError("subspace bases must be rank-two tensors")
    if left_basis.shape[0] != right_basis.shape[0]:
        raise ValueError("subspace bases must use the same ambient dimension")
    left_q = _orthonormal_columns(left_basis, tolerance=tolerance)
    right_q = _orthonormal_columns(right_basis, tolerance=tolerance)
    if left_q.shape[1] == 0 or right_q.shape[1] == 0:
        return PrincipalAngles((), (), int(left_q.shape[1]), int(right_q.shape[1]))
    singular_values = torch.linalg.svdvals(left_q.T @ right_q).clamp(0.0, 1.0)
    cosines = tuple(float(value) for value in singular_values)
    angles = tuple(math.acos(value) for value in cosines)
    return PrincipalAngles(angles, cosines, int(left_q.shape[1]), int(right_q.shape[1]))


def low_rank_subspace_diagnostics(
    *,
    left_left_factor: torch.Tensor,
    left_right_factor: torch.Tensor,
    right_left_factor: torch.Tensor,
    right_right_factor: torch.Tensor,
) -> LowRankSubspaceDiagnostics:
    """Compare the column and row spaces of ``B @ A`` low-rank deltas."""

    for left_factor, right_factor, label in (
        (left_left_factor, left_right_factor, "left patch"),
        (right_left_factor, right_right_factor, "right patch"),
    ):
        if left_factor.ndim != 2 or right_factor.ndim != 2:
            raise ValueError(f"{label} factors must be rank-two tensors")
        if left_factor.shape[1] != right_factor.shape[0]:
            raise ValueError(f"{label} inner low-rank dimensions do not match")
    if left_left_factor.shape[0] != right_left_factor.shape[0]:
        raise ValueError("low-rank deltas have different output dimensions")
    if left_right_factor.shape[1] != right_right_factor.shape[1]:
        raise ValueError("low-rank deltas have different input dimensions")
    return LowRankSubspaceDiagnostics(
        column_space=principal_angles(left_left_factor, right_left_factor),
        row_space=principal_angles(left_right_factor.T, right_right_factor.T),
    )


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.shape != right.shape:
        raise ValueError("cosine-similarity tensors must have equal shapes")
    left_flat = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right_flat = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = float(torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat))
    if denominator == 0.0:
        return None
    return float(torch.dot(left_flat, right_flat)) / denominator


def output_interaction_residual(
    *, base: torch.Tensor, left: torch.Tensor, right: torch.Tensor, composed: torch.Tensor
) -> torch.Tensor:
    if not (base.shape == left.shape == right.shape == composed.shape):
        raise ValueError("interaction outputs must have equal shapes")
    return composed - left - right + base


def contract_margin_interaction(
    *, base_margin: float, left_margin: float, right_margin: float, composed_margin: float
) -> float:
    """Return ``m(p+q) - m(p) - m(q) + m(base)``."""

    values = (base_margin, left_margin, right_margin, composed_margin)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("contract margins must be finite")
    return composed_margin - left_margin - right_margin + base_margin
