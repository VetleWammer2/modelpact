"""Canonical patch-subset representations and deterministic enumeration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

PatchSubset = tuple[str, ...]
InclusionVector = tuple[int, ...]


def canonical_subset(
    members: Iterable[str], *, universe: Sequence[str] | None = None
) -> PatchSubset:
    result = tuple(sorted(set(members)))
    if universe is not None:
        unknown = sorted(set(result) - set(universe))
        if unknown:
            raise ValueError(f"subset contains unknown patches: {unknown}")
    return result


def validate_patch_universe(patches: Sequence[str]) -> tuple[str, ...]:
    if any(not patch for patch in patches):
        raise ValueError("patch identities must not be empty")
    if len(patches) != len(set(patches)):
        raise ValueError("patch universe contains duplicate identities")
    return tuple(sorted(patches))


def enumerate_subsets(
    patches: Sequence[str],
    *,
    minimum_order: int = 1,
    maximum_order: int | None = None,
    include_empty: bool = False,
) -> tuple[PatchSubset, ...]:
    """Enumerate subsets by cardinality and then lexicographically."""

    universe = validate_patch_universe(patches)
    if minimum_order < 0:
        raise ValueError("minimum_order must be non-negative")
    upper = len(universe) if maximum_order is None else maximum_order
    if upper < 0 or upper > len(universe):
        raise ValueError("maximum_order must be between zero and the patch count")
    if minimum_order > upper:
        return ((),) if include_empty else ()
    output: list[PatchSubset] = []
    if include_empty:
        output.append(())
    for order in range(max(1, minimum_order), upper + 1):
        output.extend(combinations(universe, order))
    return tuple(output)


def subset_to_vector(subset: Iterable[str], patches: Sequence[str]) -> InclusionVector:
    universe = validate_patch_universe(patches)
    canonical = canonical_subset(subset, universe=universe)
    selected = set(canonical)
    return tuple(1 if patch in selected else 0 for patch in universe)


def vector_to_subset(vector: Sequence[int], patches: Sequence[str]) -> PatchSubset:
    universe = validate_patch_universe(patches)
    if len(vector) != len(universe):
        raise ValueError("inclusion vector length does not match patch universe")
    if any(value not in (0, 1) for value in vector):
        raise ValueError("inclusion vectors must be binary")
    return tuple(patch for patch, included in zip(universe, vector, strict=True) if included)


def hamming_distance(left: InclusionVector, right: InclusionVector) -> int:
    if len(left) != len(right):
        raise ValueError("inclusion vectors must have equal length")
    return sum(
        left_value != right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
