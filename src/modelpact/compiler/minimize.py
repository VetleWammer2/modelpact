"""Executed module and rank minimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from modelpact.checkpoints.safetensors import tensor_content_hash
from modelpact.compiler.gradient_basis import low_rank_factors
from modelpact.status import MinimalityClaim
from modelpact.util.hashing import hash_canonical


@dataclass(frozen=True, slots=True)
class MinimizationCandidate:
    operation: str
    candidate_id: str
    active_modules: tuple[str, ...]
    ranks: dict[str, int]
    passed: bool


@dataclass(frozen=True, slots=True)
class PatchMinimizationResult:
    deltas: dict[str, Tensor]
    factors: dict[str, tuple[Tensor, Tensor]]
    candidates: tuple[MinimizationCandidate, ...]
    claims: tuple[MinimalityClaim, ...]
    verification_budget_used: int


Verifier = Callable[[dict[str, Tensor]], bool]


def _representation_ranks(
    deltas: dict[str, Tensor], factors: dict[str, tuple[Tensor, Tensor]]
) -> dict[str, int]:
    return {
        name: (
            int(factors[name][0].shape[1])
            if name in factors
            else (min(value.shape) if value.ndim == 2 else 0)
        )
        for name, value in deltas.items()
    }


def _candidate_id(deltas: dict[str, Tensor]) -> str:
    return hash_canonical(
        {name: tensor_content_hash(value.detach().cpu()) for name, value in sorted(deltas.items())}
    )


def minimize_patch(
    deltas: dict[str, Tensor],
    verifier: Verifier,
    *,
    verification_budget: int = 100,
    seed: int = 0,
    initial_factors: dict[str, tuple[Tensor, Tensor]] | None = None,
) -> PatchMinimizationResult:
    if verification_budget <= 0:
        raise ValueError("verification budget must be positive")
    current = {name: value.detach().clone() for name, value in sorted(deltas.items())}
    current_factors: dict[str, tuple[Tensor, Tensor]] = {}
    if initial_factors is not None:
        if set(initial_factors) != set(current):
            raise ValueError("initial factors must cover exactly the minimized deltas")
        for name, (left, right) in sorted(initial_factors.items()):
            materialized = left.detach() @ right.detach()
            if not torch.equal(materialized, current[name]):
                raise ValueError(f"initial factors do not materialize delta {name!r}")
            current_factors[name] = (left.detach().clone(), right.detach().clone())
    initial_passed = verifier(current)
    candidates: list[MinimizationCandidate] = [
        MinimizationCandidate(
            "verify:initial",
            _candidate_id(current),
            tuple(current),
            _representation_ranks(current, current_factors),
            initial_passed,
        )
    ]
    used = 1
    if not initial_passed:
        raise ValueError("cannot minimize a patch that fails its executed verifier")
    one_minimal = not current
    # Greedy one-removal tests establish one-minimality only if every retained
    # module removal was executed and failed.
    changed = True
    while current and changed and used < verification_budget:
        changed = False
        completed_pass = True
        for name in tuple(current):
            if used >= verification_budget:
                completed_pass = False
                break
            candidate = {key: value for key, value in current.items() if key != name}
            candidate_factors = {
                key: value for key, value in current_factors.items() if key != name
            }
            passed = verifier(candidate)
            used += 1
            candidates.append(
                MinimizationCandidate(
                    f"remove:{name}",
                    _candidate_id(candidate),
                    tuple(candidate),
                    _representation_ranks(candidate, candidate_factors),
                    passed,
                )
            )
            if passed:
                current = candidate
                current_factors = candidate_factors
                changed = True
                completed_pass = False
                if not current:
                    one_minimal = True
                break
        if not changed and completed_pass:
            one_minimal = True
    rank_local = True
    for name in tuple(current):
        matrix = current[name]
        # Vectors have no matrix rank to reduce. They still participate in the
        # executed module-removal phase above.
        if matrix.ndim != 2:
            continue
        full_rank = int(torch.linalg.matrix_rank(matrix.to(torch.float64)).item())
        if name in current_factors:
            full_rank = min(full_rank, current_factors[name][0].shape[1])
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
                    _candidate_id(candidate),
                    tuple(candidate),
                    {
                        **_representation_ranks(candidate, current_factors),
                        name: rank,
                    },
                    passed,
                )
            )
            if passed:
                current = candidate
                current_factors[name] = (left.detach().clone(), right.detach().clone())
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
    for name, matrix in sorted(current.items()):
        if matrix.ndim == 2 and name not in current_factors:
            current_factors[name] = (
                matrix.detach().clone(),
                torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device),
            )
    return PatchMinimizationResult(
        current,
        current_factors,
        tuple(candidates),
        tuple(claims),
        used,
    )
