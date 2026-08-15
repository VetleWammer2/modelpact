"""Delta-debugging minimization for difference witnesses and counterexamples."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MinimizationResult:
    original: str
    minimized: str
    evaluations: int
    accepted_reductions: int


def ddmin_items(items: Sequence[str], predicate: Callable[[Sequence[str]], bool]) -> tuple[tuple[str, ...], int, int]:
    current = tuple(items)
    if not current:
        return current, 0, 0
    evaluations = 0
    accepted = 0
    granularity = 2
    while len(current) >= 2:
        chunk_size = max(1, len(current) // granularity)
        reduced = False
        for start in range(0, len(current), chunk_size):
            candidate = current[:start] + current[start + chunk_size :]
            if not candidate:
                continue
            evaluations += 1
            if predicate(candidate):
                current = candidate
                accepted += 1
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return current, evaluations, accepted


def minimize_prompt(prompt: str, preserves_failure: Callable[[str], bool]) -> MinimizationResult:
    evaluations = 0
    accepted = 0
    clauses = tuple(item for item in re.split(r"(?<=[.!?;])\s+|\n+", prompt) if item)

    def clause_predicate(items: Sequence[str]) -> bool:
        return preserves_failure(" ".join(items))

    clauses, count, reductions = ddmin_items(clauses, clause_predicate)
    evaluations += count
    accepted += reductions
    clause_minimum = " ".join(clauses)
    tokens = tuple(re.findall(r"\S+", clause_minimum))

    def token_predicate(items: Sequence[str]) -> bool:
        return preserves_failure(" ".join(items))

    tokens, count, reductions = ddmin_items(tokens, token_predicate)
    return MinimizationResult(prompt, " ".join(tokens), evaluations + count, accepted + reductions)

