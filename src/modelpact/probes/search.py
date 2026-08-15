"""Budgeted deterministic search over local prompt mutations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from modelpact.probes.mutations import (
    DEFAULT_MUTATION_CONFIG,
    Mutation,
    MutationConfig,
    mutate_prompt,
)


@dataclass(frozen=True, slots=True)
class SearchScore:
    divergence: float
    novelty: float
    coverage: float
    complexity: float

    def objective(
        self, *, novelty_weight: float, coverage_weight: float, complexity_weight: float
    ) -> float:
        return (
            self.divergence
            + novelty_weight * self.novelty
            + coverage_weight * self.coverage
            - complexity_weight * self.complexity
        )


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    prompt: str
    mutation: Mutation | None
    score: SearchScore


@dataclass(frozen=True, slots=True)
class SearchConfig:
    budget: int = 256
    seed: int = 0
    novelty_weight: float = 0.1
    coverage_weight: float = 0.1
    complexity_weight: float = 0.001


DEFAULT_SEARCH_CONFIG = SearchConfig()


def search_prompts(
    seeds: Iterable[str],
    evaluator: Callable[[str], tuple[float, float, float]],
    *,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    mutation_config: MutationConfig = DEFAULT_MUTATION_CONFIG,
) -> tuple[SearchCandidate, ...]:
    """Evaluate seed and mutated prompts, returning stable objective order.

    The evaluator returns ``(divergence, novelty, coverage)``. Model execution,
    rather than the search heuristic, determines whether a prompt is a witness.
    """

    ordered_seeds = tuple(dict.fromkeys(seeds))
    if not ordered_seeds:
        raise ValueError("prompt search requires at least one seed")
    if len(ordered_seeds) > config.budget:
        raise ValueError("search budget must cover every declared seed probe")
    candidates: list[SearchCandidate] = []
    seen: set[str] = set()
    # Every fixed seed is executed before fuzzing. This prevents an early
    # seed's mutation fan-out from consuming the budget for later seed probes.
    for prompt in ordered_seeds:
        divergence, novelty, coverage = evaluator(prompt)
        score = SearchScore(
            divergence,
            novelty,
            coverage,
            float(len(prompt.encode("utf-8"))),
        )
        candidates.append(SearchCandidate(prompt, None, score))
        seen.add(prompt)
    mutation_queues = tuple(
        mutate_prompt(prompt, config=mutation_config, seed=config.seed + seed_index)
        for seed_index, prompt in enumerate(ordered_seeds)
    )
    maximum_mutations = max((len(queue) for queue in mutation_queues), default=0)
    for mutation_index in range(maximum_mutations):
        for _prompt, queue in zip(ordered_seeds, mutation_queues, strict=True):
            if mutation_index >= len(queue):
                continue
            mutation = queue[mutation_index]
            candidate_prompt = mutation.mutated
            if candidate_prompt in seen or len(candidates) >= config.budget:
                continue
            seen.add(candidate_prompt)
            divergence, novelty, coverage = evaluator(candidate_prompt)
            score = SearchScore(
                divergence, novelty, coverage, float(len(candidate_prompt.encode("utf-8")))
            )
            candidates.append(SearchCandidate(candidate_prompt, mutation, score))
        if len(candidates) >= config.budget:
            break
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item.score.objective(
                    novelty_weight=config.novelty_weight,
                    coverage_weight=config.coverage_weight,
                    complexity_weight=config.complexity_weight,
                ),
                item.prompt,
            ),
        )
    )
