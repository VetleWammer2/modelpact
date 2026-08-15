"""Executed module and rank minimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from modelpact.compiler.gradient_basis import low_rank_factors
from modelpact.status import MinimalityClaim


@dataclass(frozen=True, slots=True)
class MinimizationCandidate:
    operation: str
    active_modules: tuple[str, ...]
    ranks: dict[str, int]
    passed: bool


@dataclass(frozen=True, slots=True)
class PatchMinimizationResult:
    deltas: dict[str, Tensor]
    candidates: tuple[MinimizationCandidate, ...]
    claims: tuple[MinimalityClaim, ...]
    verification_budget_used: int


Verifier = Callable[[dict[str, Tensor]], bool]


def minimize_patch(
    deltas: dict[str, Tensor],
    verifier: Verifier,
    *,
    verification_budget: int = 100,
    seed: int = 0,
) -> PatchMinimizationResult:
    if verification_budget <= 0:
        raise ValueError("verification budget must be positive")
    current = {name: value.detach().clone() for name, value in sorted(deltas.items())}
    candidates: list[MinimizationCandidate] = []
    used = 0
    one_minimal = True
    # Greedy one-removal tests establish one-minimality only if every retained
    # module removal was executed and failed.
    changed = True
    while changed and used < verification_budget:
        changed = False
        for name in tuple(current):
            if used >= verification_budget:
                one_minimal = False
                break
            candidate = {key: value for key, value in current.items() if key != name}
            passed = verifier(candidate)
            used += 1
            candidates.append(
                MinimizationCandidate(
                    f"remove:{name}",
                    tuple(candidate),
                    {key: min(candidate[key].shape) for key in candidate},
                    passed,
                )
            )
            if passed:
                current = candidate
                changed = True
                break
    rank_local = True
    for name in tuple(current):
        matrix = current[name]
        full_rank = int(torch.linalg.matrix_rank(matrix.to(torch.float64)).item())
        for rank in range(1, max(1, full_rank)):
            if used >= verification_budget:
                rank_local = False
                break
            left, right = low_rank_factors(matrix, rank=rank, seed=seed)
            candidate = dict(current)
            candidate[name] = left @ right
            passed = verifier(candidate)
            used += 1
            candidates.append(
                MinimizationCandidate(
                    f"rank:{name}:{rank}",
                    tuple(candidate),
                    {key: rank if key == name else min(candidate[key].shape) for key in candidate},
                    passed,
                )
            )
            if passed:
                current = candidate
                break
    claims: list[MinimalityClaim] = []
    if one_minimal:
        claims.append(MinimalityClaim.MODULE_ONE_MINIMAL)
    if rank_local:
        claims.append(MinimalityClaim.RANK_LOCAL_MINIMUM)
    if used >= verification_budget and (not one_minimal or not rank_local):
        claims.append(MinimalityClaim.BUDGET_MINIMAL)
    if not claims:
        claims.append(MinimalityClaim.UNMINIMIZED)
    return PatchMinimizationResult(current, tuple(candidates), tuple(claims), used)
