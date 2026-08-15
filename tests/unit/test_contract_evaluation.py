from __future__ import annotations

import math

import pytest
import torch

from modelpact.contracts import (
    AssertionType,
    CompileObjective,
    EvaluationRecord,
    ObjectiveInputs,
    ObjectiveType,
    StatisticsPolicy,
    VerificationAssertion,
    evaluate_assertion,
    evaluate_objective,
)
from modelpact.status import VerificationOutcome


def objective(kind: ObjectiveType, **options: object) -> CompileObjective:
    return CompileObjective(
        id=f"objective-{kind.value}",
        type=kind,
        source="train.jsonl",
        options=options,
    )


def assertion(kind: AssertionType, **options: object) -> VerificationAssertion:
    return VerificationAssertion(
        id=f"assertion-{kind.value}",
        type=kind,
        source="validation.jsonl",
        options=options,
    )


def test_all_differentiable_objectives_execute_and_preserve_gradients() -> None:
    logits = torch.randn(2, 4, 7, requires_grad=True)
    teacher = torch.randn(2, 4, 7)
    labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    preferred = torch.tensor([2.0, 1.0], requires_grad=True)
    dispreferred = torch.tensor([0.0, 0.5], requires_grad=True)
    hidden = torch.randn(2, 3, 5, requires_grad=True)
    reference_hidden = torch.randn(2, 3, 5)
    activations = torch.randn(2, 3, 5, requires_grad=True)
    direction = torch.randn(5)
    cases = (
        (
            objective(ObjectiveType.TEACHER_CROSS_ENTROPY),
            ObjectiveInputs(logits=logits, labels=labels),
        ),
        (
            objective(ObjectiveType.TEACHER_KL),
            ObjectiveInputs(logits=logits, teacher_logits=teacher),
        ),
        (
            objective(ObjectiveType.BASE_KL),
            ObjectiveInputs(logits=logits, base_logits=teacher),
        ),
        (
            objective(ObjectiveType.PREFERRED_SEQUENCE_MARGIN, margin=0.5),
            ObjectiveInputs(
                preferred_log_prob=preferred,
                dispreferred_log_prob=dispreferred,
            ),
        ),
        (
            objective(ObjectiveType.HIDDEN_STATE_MATCHING, metric="cosine"),
            ObjectiveInputs(
                hidden_states=hidden,
                reference_hidden_states=reference_hidden,
            ),
        ),
        (
            objective(ObjectiveType.ACTIVATION_DIRECTION, minimum_projection=0.2),
            ObjectiveInputs(activations=activations, direction=direction),
        ),
    )
    total = torch.zeros(())
    for spec, inputs in cases:
        result = evaluate_objective(spec, inputs)
        assert result.outcome is VerificationOutcome.PASS
        assert result.loss is not None
        assert torch.isfinite(result.loss)
        total = total + result.loss
    total.backward()
    assert logits.grad is not None
    assert hidden.grad is not None
    assert activations.grad is not None


def test_missing_objective_evidence_is_honestly_unsupported() -> None:
    result = evaluate_objective(
        objective(ObjectiveType.TEACHER_KL),
        ObjectiveInputs(logits=torch.randn(1, 2, 3)),
    )
    assert result.outcome is VerificationOutcome.UNSUPPORTED
    assert result.loss is None


@pytest.mark.parametrize(
    ("kind", "options", "record"),
    [
        (
            AssertionType.EXACT_MATCH,
            {"expected": "Hello"},
            EvaluationRecord("a", "prompt", generated_text="Hello"),
        ),
        (
            AssertionType.NORMALIZED_EXACT_MATCH,
            {"expected": "hello world"},
            EvaluationRecord("a", "prompt", generated_text="  hello   world "),
        ),
        (
            AssertionType.REGULAR_EXPRESSION,
            {"pattern": r"^value=[0-9]+$", "full_match": True},
            EvaluationRecord("a", "prompt", generated_text="value=17"),
        ),
        (
            AssertionType.JSON_PARSE,
            {},
            EvaluationRecord("a", "prompt", generated_text='{"ok": true}'),
        ),
        (
            AssertionType.JSON_SCHEMA,
            {
                "schema": {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "integer", "minimum": 0}},
                    "additionalProperties": False,
                }
            },
            EvaluationRecord("a", "prompt", generated_text='{"answer": 3}'),
        ),
        (
            AssertionType.FREE_GENERATION_MATCH,
            {"expected": "answer", "match_type": "contains"},
            EvaluationRecord("a", "prompt", generated_text="the answer is 3"),
        ),
        (
            AssertionType.SEQUENCE_MARGIN,
            {"preferred": "yes", "dispreferred": "no", "minimum_margin": 1.0},
            EvaluationRecord(
                "a",
                "prompt",
                values={"sequence_log_probabilities": {"yes": -1.0, "no": -3.0}},
            ),
        ),
        (
            AssertionType.MULTIPLE_CHOICE_MARGIN,
            {"choices": ["A", "B"], "correct_choice": "A", "minimum_margin": 0.5},
            EvaluationRecord(
                "a",
                "prompt",
                values={"choice_log_probabilities": {"A": -0.2, "B": -1.0}},
            ),
        ),
        (
            AssertionType.GENERATION_LENGTH,
            {"minimum": 2, "maximum": 4, "unit": "tokens"},
            EvaluationRecord("a", "prompt", generated_text="x", generated_token_ids=(1, 2, 3)),
        ),
    ],
)
def test_discrete_and_generation_assertions_pass(
    kind: AssertionType,
    options: dict[str, object],
    record: EvaluationRecord,
) -> None:
    result = evaluate_assertion(
        assertion(kind, **options),
        (record,),
        statistics=StatisticsPolicy(bootstrap_samples=20, bootstrap_seed=2),
    )
    assert result.outcome is VerificationOutcome.PASS
    assert result.margin is not None and result.margin >= 0
    assert result.prompt_metrics[0].prompt_hash.startswith("sha256:")
    assert result.confidence_interval is not None


def causal_record() -> EvaluationRecord:
    logits = torch.full((3, 5), -3.0)
    ids = torch.tensor([0, 1, 2, 3])
    logits[0, 1] = 3.0
    logits[1, 2] = 3.0
    logits[2, 3] = 3.0
    return EvaluationRecord("numeric", "abc", logits=logits, input_ids=ids)


def test_token_and_sequence_log_probability_use_real_logits() -> None:
    record = causal_record()
    token = evaluate_assertion(
        assertion(AssertionType.TOKEN_LOG_PROBABILITY, position=3, minimum=-0.1),
        (record,),
    )
    sequence = evaluate_assertion(
        assertion(AssertionType.SEQUENCE_LOG_PROBABILITY, normalize=True, minimum=-0.1),
        (record,),
    )
    assert token.outcome is VerificationOutcome.PASS
    assert sequence.outcome is VerificationOutcome.PASS
    assert token.value is not None and token.value < 0


def test_reference_and_base_kl_use_distributions_not_argmax_only() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.5, -0.5]])
    changed = logits + torch.tensor([[0.01, -0.01], [0.0, 0.0]])
    record = EvaluationRecord(
        "kl",
        "p",
        logits=changed,
        input_ids=torch.tensor([0, 1]),
        reference_logits=logits,
        base_logits=logits,
    )
    for kind in (AssertionType.REFERENCE_KL, AssertionType.BASE_KL):
        result = evaluate_assertion(assertion(kind, maximum_mean=0.01), (record,))
        assert result.outcome is VerificationOutcome.PASS
        assert result.value is not None and 0 <= result.value < 0.01


def test_perplexity_is_computed_from_next_token_cross_entropy() -> None:
    result = evaluate_assertion(
        assertion(AssertionType.PERPLEXITY, maximum_mean=1.2),
        (causal_record(),),
    )
    assert result.outcome is VerificationOutcome.PASS
    assert result.value is not None and 1.0 <= result.value < 1.2


def test_assertion_failure_records_prompt_level_margin() -> None:
    result = evaluate_assertion(
        assertion(AssertionType.EXACT_MATCH, expected="wanted"),
        (EvaluationRecord("failure", "prompt", generated_text="observed"),),
    )
    assert result.outcome is VerificationOutcome.FAIL
    assert result.margin == -1.0
    assert result.prompt_metrics[0].outcome is VerificationOutcome.FAIL
    assert result.prompt_metrics[0].output_hash is not None


def test_missing_scoring_evidence_and_unsupported_schema_are_not_success() -> None:
    missing = evaluate_assertion(
        assertion(AssertionType.REFERENCE_KL, maximum_mean=0.1),
        (EvaluationRecord("x", "p"),),
    )
    unsupported_schema = evaluate_assertion(
        assertion(AssertionType.JSON_SCHEMA, schema={"$ref": "other.json"}),
        (EvaluationRecord("y", "p", generated_text="{}"),),
    )
    assert missing.outcome is VerificationOutcome.UNSUPPORTED
    assert unsupported_schema.outcome is VerificationOutcome.UNSUPPORTED


def test_nonfinite_runtime_metric_is_inconclusive() -> None:
    result = evaluate_assertion(
        assertion(AssertionType.PERPLEXITY, maximum_mean=2.0),
        (EvaluationRecord("x", "p", values={"perplexity": math.nan}),),
    )
    assert result.outcome is VerificationOutcome.INCONCLUSIVE
