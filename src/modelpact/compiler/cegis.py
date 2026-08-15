"""Counterexample-guided recompilation with explicit search budgets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from modelpact.compiler.result import CompilationResult

ExampleT = TypeVar("ExampleT")


class CEGISStop(StrEnum):
    NO_COUNTEREXAMPLE_WITHIN_BUDGET = "NO_COUNTEREXAMPLE_WITHIN_BUDGET"
    MAXIMUM_ROUNDS = "MAXIMUM_ROUNDS"
    INFEASIBLE = "INFEASIBLE"
    RESOURCE_BUDGET_EXHAUSTED = "RESOURCE_BUDGET_EXHAUSTED"


@dataclass(frozen=True, slots=True)
class Counterexample(Generic[ExampleT]):
    example: ExampleT
    domain: str
    margin: float
    minimized: bool
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CEGISRound(Generic[ExampleT]):
    round_index: int
    target_counterexamples: tuple[Counterexample[ExampleT], ...]
    guard_counterexamples: tuple[Counterexample[ExampleT], ...]
    search_budget: int
    compilation_feasible: bool


@dataclass(frozen=True, slots=True)
class CEGISResult(Generic[ExampleT]):
    candidate: CompilationResult
    rounds: tuple[CEGISRound[ExampleT], ...]
    stop_reason: CEGISStop
    working_target_examples: tuple[ExampleT, ...]
    working_guard_examples: tuple[ExampleT, ...]


CompilerCallback = Callable[[tuple[ExampleT, ...], tuple[ExampleT, ...]], CompilationResult]
SearchCallback = Callable[[CompilationResult, int], tuple[Counterexample[ExampleT], ...]]


def run_cegis(
    target_examples: tuple[ExampleT, ...],
    guard_examples: tuple[ExampleT, ...],
    *,
    compile_candidate: CompilerCallback[ExampleT],
    search_targets: SearchCallback[ExampleT],
    search_guards: SearchCallback[ExampleT],
    maximum_rounds: int = 5,
    search_budget_per_round: int = 128,
) -> CEGISResult[ExampleT]:
    if maximum_rounds <= 0 or search_budget_per_round <= 0:
        raise ValueError("CEGIS round and search budgets must be positive")
    working_targets = list(target_examples)
    working_guards = list(guard_examples)
    seen_targets = set(target_examples)
    seen_guards = set(guard_examples)
    history: list[CEGISRound[ExampleT]] = []
    candidate = compile_candidate(tuple(working_targets), tuple(working_guards))
    for round_index in range(maximum_rounds):
        if not candidate.feasible:
            return CEGISResult(candidate, tuple(history), CEGISStop.INFEASIBLE, tuple(working_targets), tuple(working_guards))
        target_found = search_targets(candidate, search_budget_per_round)
        guard_found = search_guards(candidate, search_budget_per_round)
        new_targets = tuple(item for item in target_found if item.example not in seen_targets)
        new_guards = tuple(item for item in guard_found if item.example not in seen_guards)
        history.append(CEGISRound(round_index, new_targets, new_guards, search_budget_per_round, candidate.feasible))
        if not new_targets and not new_guards:
            return CEGISResult(
                candidate,
                tuple(history),
                CEGISStop.NO_COUNTEREXAMPLE_WITHIN_BUDGET,
                tuple(working_targets),
                tuple(working_guards),
            )
        for item in new_targets:
            seen_targets.add(item.example)
            working_targets.append(item.example)
        for item in new_guards:
            seen_guards.add(item.example)
            working_guards.append(item.example)
        candidate = compile_candidate(tuple(working_targets), tuple(working_guards))
    return CEGISResult(candidate, tuple(history), CEGISStop.MAXIMUM_ROUNDS, tuple(working_targets), tuple(working_guards))

