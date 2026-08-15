"""Budgeted deterministic search over local prompt mutations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from modelpact.probes.mutations import Mutation, MutationConfig, mutate_prompt


@dataclass(frozen=True, slots=True)
class SearchScore:
    divergence: float
    novelty: float
    coverage: float
    complexity: float

    def objective(self, *, novelty_weight: float, coverage_weight: float, complexity_weight: float) -> float:
        return self.divergence + novelty_weight * self.novelty + coverage_weight * self.coverage - complexity_weight * self.complexity


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


def search_prompts(
    seeds: Iterable[str],
    evaluator: Callable[[str], tuple[float, float, float]],
    *,
    config: SearchConfig = SearchConfig(),
    mutation_config: MutationConfig = MutationConfig(),
) -> tuple[SearchCandidate, ...]:
    """Evaluate seed and mutated prompts, returning stable objective order.

    The evaluator returns ``(divergence, novelty, coverage)``. Model execution,
    rather than the search heuristic, determines whether a prompt is a witness.
    """

    candidates: list[SearchCandidate] = []
    seen: set[str] = set()
    for seed_index, prompt in enumerate(seeds):
        queue: tuple[Mutation | None, ...] = (None, *mutate_prompt(prompt, config=mutation_config, seed=config.seed + seed_index))
        for mutation in queue:
            candidate_prompt = prompt if mutation is None else mutation.mutated
            if candidate_prompt in seen or len(candidates) >= config.budget:
                continue
            seen.add(candidate_prompt)
            divergence, novelty, coverage = evaluator(candidate_prompt)
            score = SearchScore(divergence, novelty, coverage, float(len(candidate_prompt.encode("utf-8"))))
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

