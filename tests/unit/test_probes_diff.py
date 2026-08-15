from __future__ import annotations

from modelpact.diff.cluster import deterministic_agglomerative
from modelpact.diff.witnesses import DifferenceWitness
from modelpact.probes.minimize import minimize_prompt
from modelpact.probes.mutations import MutationOperator, mutate_prompt


def _witness(prompt: str, distance: float) -> DifferenceWitness:
    return DifferenceWitness.create(
        original_input=prompt,
        minimized_input=prompt,
        divergence_metrics={"symmetric_kl": distance},
        base_output={"token": 0},
        target_output={"token": 1},
        activation_fingerprint=(distance, 0.0),
    )


def test_mutation_is_deterministic_and_covers_required_operators() -> None:
    prompt = "Aster has 7 items; return them. Keep format."
    left = mutate_prompt(prompt, seed=41)
    right = mutate_prompt(prompt, seed=41)
    assert left == right
    operators = {item.operator for item in left}
    assert MutationOperator.ENTITY_SUBSTITUTION in operators
    assert MutationOperator.NUMBER_SUBSTITUTION in operators
    assert MutationOperator.INSTRUCTION_ORDER in operators
    assert MutationOperator.ROLE_WRAPPER in operators


def test_prompt_ddmin_executes_failure_oracle() -> None:
    result = minimize_prompt(
        "irrelevant words trigger more noise", lambda value: "trigger" in value
    )
    assert result.minimized == "trigger"
    assert result.evaluations > 0


def test_clustering_is_deterministic() -> None:
    witnesses = (_witness("a", 0.01), _witness("b", 0.02), _witness("c", 10.0))
    first = deterministic_agglomerative(witnesses, maximum_clusters=2, distance_threshold=0.5)
    second = deterministic_agglomerative(
        tuple(reversed(witnesses)), maximum_clusters=2, distance_threshold=0.5
    )
    assert first == second
    assert len(first) == 2
