"""Deterministic initial designs for combinatorial patch audits."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from modelpact.audit.subsets import PatchSubset, canonical_subset, enumerate_subsets


@dataclass(frozen=True, slots=True)
class InitialDesignConfig:
    include_pairs: bool = True
    balanced_random_subsets: int = 0
    maximum_order: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.balanced_random_subsets < 0:
            raise ValueError("balanced_random_subsets must be non-negative")
        if self.maximum_order is not None and self.maximum_order < 1:
            raise ValueError("maximum_order must be positive when provided")


def _balanced_random_design(
    patches: tuple[str, ...],
    *,
    count: int,
    maximum_order: int,
    seed: int,
    excluded: set[PatchSubset],
) -> tuple[PatchSubset, ...]:
    candidates = list(
        enumerate_subsets(
            patches,
            minimum_order=2,
            maximum_order=maximum_order,
        )
    )
    candidates = [candidate for candidate in candidates if candidate not in excluded]
    generator = random.Random(seed)  # noqa: S311 -- deterministic experimental design
    generator.shuffle(candidates)
    exposure = dict.fromkeys(patches, 0)
    selected: list[PatchSubset] = []
    while candidates and len(selected) < count:
        # Favor candidates containing currently under-exposed patches.  Random
        # shuffle above supplies a deterministic seed-dependent final tie break.
        position, best = max(
            enumerate(candidates),
            key=lambda item: (
                -sum(exposure[patch] for patch in item[1]),
                len(item[1]),
                -item[0],
            ),
        )
        selected.append(best)
        for patch in best:
            exposure[patch] += 1
        candidates.pop(position)
    return tuple(selected)


def initial_design(
    patches: Sequence[str],
    *,
    config: InitialDesignConfig,
    user_requested: Sequence[Sequence[str]] = (),
    high_risk: Sequence[Sequence[str]] = (),
) -> tuple[PatchSubset, ...]:
    universe = tuple(sorted(patches))
    maximum_order = config.maximum_order or len(universe)
    if maximum_order > len(universe):
        raise ValueError("maximum_order exceeds patch count")
    design: set[PatchSubset] = {(patch,) for patch in universe}
    if config.include_pairs and maximum_order >= 2:
        design.update(combinations(universe, 2))
    for requested in (*user_requested, *high_risk):
        subset = canonical_subset(requested, universe=universe)
        if not subset:
            continue
        if len(subset) > maximum_order:
            raise ValueError("requested subset exceeds maximum_order")
        design.add(subset)
    random_design = _balanced_random_design(
        universe,
        count=config.balanced_random_subsets,
        maximum_order=maximum_order,
        seed=config.seed,
        excluded=design,
    )
    design.update(random_design)
    return tuple(sorted(design, key=lambda subset: (len(subset), subset)))
