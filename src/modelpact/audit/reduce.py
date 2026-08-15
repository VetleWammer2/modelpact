"""Executed delta debugging for failing patch subsets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from modelpact.audit.subsets import PatchSubset, canonical_subset


class FailureOracle(Protocol):
    def __call__(self, subset: PatchSubset) -> bool: ...


class ReductionBudgetExhausted(RuntimeError):
    """Raised by a budget-aware oracle before an unapproved execution."""


@dataclass(frozen=True, slots=True)
class ReductionResult:
    original: PatchSubset
    reduced: PatchSubset
    tested_candidates: tuple[PatchSubset, ...]
    one_minimal: bool
    budget_exhausted: bool


def _partitions(items: PatchSubset, count: int) -> tuple[PatchSubset, ...]:
    size, remainder = divmod(len(items), count)
    output: list[PatchSubset] = []
    start = 0
    for index in range(count):
        width = size + (1 if index < remainder else 0)
        output.append(items[start : start + width])
        start += width
    return tuple(part for part in output if part)


def ddmin_failing_subset(
    failing_subset: Sequence[str],
    *,
    oracle: FailureOracle,
    initial_known_failing: bool = True,
    maximum_evaluations: int | None = None,
) -> ReductionResult:
    """Reduce a witnessed failure using deterministic ddmin executions."""

    current = canonical_subset(failing_subset)
    if not current:
        raise ValueError("a failing subset must not be empty")
    if maximum_evaluations is not None and maximum_evaluations < 0:
        raise ValueError("maximum_evaluations must be non-negative")
    tested: list[PatchSubset] = []
    calls = 0

    def evaluate(candidate: PatchSubset) -> bool:
        nonlocal calls
        if maximum_evaluations is not None and calls >= maximum_evaluations:
            raise ReductionBudgetExhausted
        calls += 1
        tested.append(candidate)
        return oracle(candidate)

    try:
        if not initial_known_failing and not evaluate(current):
            raise ValueError("the supplied subset does not fail")
        granularity = min(2, len(current))
        while len(current) >= 2:
            parts = _partitions(current, granularity)
            reduced = False
            for part in parts:
                part_set = set(part)
                complement = tuple(item for item in current if item not in part_set)
                if complement and evaluate(complement):
                    current = complement
                    granularity = max(granularity - 1, 2)
                    reduced = True
                    break
            if reduced:
                continue
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    except ReductionBudgetExhausted:
        return ReductionResult(
            original=canonical_subset(failing_subset),
            reduced=current,
            tested_candidates=tuple(tested),
            one_minimal=False,
            budget_exhausted=True,
        )
    return ReductionResult(
        original=canonical_subset(failing_subset),
        reduced=current,
        tested_candidates=tuple(tested),
        one_minimal=True,
        budget_exhausted=False,
    )
