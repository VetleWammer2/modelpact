from __future__ import annotations

import json

import pytest

from modelpact.contracts import (
    AssertionType,
    ContractLimits,
    ContractResourceLimitError,
    ContractSyntaxError,
    ContractValidationError,
    ObjectiveType,
    canonical_contract_json,
    check_static_contracts,
    loads_contract,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def contract_text(
    *,
    contract_id: str = "json-mode",
    tokenizer_hash: str = HASH_A,
    base_signature: str = HASH_A,
    target: str | None = None,
) -> str:
    target_block = (
        target
        or """
    - id: exact-json
      type: exact_match
      source: probes/validation.jsonl
      expected: '{"answer": 3}'
      minimum_pass_rate: 0.9
"""
    )
    return f"""
schema_version: 1
id: {contract_id}
contract_version: 2
description: deterministic JSON behavior
model_requirements:
  tokenizer_hash: {tokenizer_hash}
  base_signature: {base_signature}
  output_semantics: causal_lm
compile:
  objectives:
    - id: imitate
      type: teacher_cross_entropy
      source: probes/train.jsonl
      weight: 1.5
      causal_shift: true
verify:
  targets:
{target_block}
  guards:
    - id: preserve-base
      type: base_kl
      source: guards/validation.jsonl
      maximum_mean: 0.02
      maximum_item: 0.1
holdout:
  sealed: true
  targets: holdout/targets.jsonl
  guards: holdout/guards.jsonl
  unseal_policy: final_candidate_only
statistics:
  confidence_level: 0.95
  bootstrap_samples: 20
  bootstrap_seed: 13
generation:
  mode: greedy
  max_new_tokens: 32
"""


def test_contract_round_trip_and_stable_hash() -> None:
    contract = loads_contract(contract_text())
    encoded = canonical_contract_json(contract)
    reparsed = loads_contract(encoded, format="json")
    assert reparsed.to_dict() == contract.to_dict()
    assert reparsed.contract_id == contract.contract_id
    assert contract.objectives[0].type is ObjectiveType.TEACHER_CROSS_ENTROPY
    assert contract.targets[0].type is AssertionType.EXACT_MATCH


def test_contract_hash_ignores_input_key_order() -> None:
    first = loads_contract(contract_text())
    value = json.loads(canonical_contract_json(first))
    reversed_value = dict(reversed(list(value.items())))
    second = loads_contract(json.dumps(reversed_value), format="json")
    assert first.contract_id == second.contract_id


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("surprise: true\n", "unknown field"),
        ("schema_version: 2\n", "schema_version"),
    ],
)
def test_contract_rejects_unknown_or_future_schema(fragment: str, message: str) -> None:
    text = contract_text()
    if fragment.startswith("schema_version"):
        text = text.replace("schema_version: 1", fragment.strip())
    else:
        text += fragment
    with pytest.raises(ContractValidationError, match=message):
        loads_contract(text)


def test_duplicate_json_and_yaml_keys_are_rejected() -> None:
    with pytest.raises(ContractSyntaxError, match="duplicate JSON"):
        loads_contract('{"schema_version":1,"schema_version":1}', format="json")
    with pytest.raises(ContractSyntaxError, match="duplicate key"):
        loads_contract("schema_version: 1\nschema_version: 1\n")


def test_yaml_aliases_and_explicit_tags_are_rejected() -> None:
    with pytest.raises(ContractSyntaxError, match="anchors"):
        loads_contract("schema_version: &v 1\nid: *v\n")
    with pytest.raises(ContractSyntaxError, match="explicit tags"):
        loads_contract("schema_version: !!int 1\n")


def test_resource_limits_are_enforced_before_schema() -> None:
    nested: object = 1
    for _ in range(8):
        nested = [nested]
    with pytest.raises(ContractResourceLimitError, match="nesting depth"):
        loads_contract(
            json.dumps(nested),
            format="json",
            limits=ContractLimits(max_depth=3),
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "minimum_pass_rate: 1.1",
        "maximum_mean: -0.1",
        "maximum_quantile: {q: 0, value: 1}",
    ],
)
def test_impossible_numeric_thresholds_are_rejected(replacement: str) -> None:
    text = contract_text()
    if replacement.startswith("minimum_pass_rate"):
        text = text.replace("minimum_pass_rate: 0.9", replacement)
    else:
        text = text.replace("maximum_mean: 0.02", replacement)
    with pytest.raises(ContractValidationError):
        loads_contract(text)


def test_unknown_objective_and_assertion_scorers_are_rejected() -> None:
    with pytest.raises(ContractValidationError, match="unknown objective"):
        loads_contract(contract_text().replace("teacher_cross_entropy", "python_callback"))
    with pytest.raises(ContractValidationError, match="unknown assertion"):
        loads_contract(contract_text().replace("exact_match", "llm_as_judge"))


def test_holdout_cannot_overlap_visible_data() -> None:
    text = contract_text().replace(
        "targets: holdout/targets.jsonl", "targets: probes/validation.jsonl"
    )
    with pytest.raises(ContractValidationError, match="holdout sources"):
        loads_contract(text)


def test_mandatory_type_taxonomy_is_complete() -> None:
    assert {item.value for item in ObjectiveType} == {
        "teacher_cross_entropy",
        "teacher_kl",
        "preferred_sequence_margin",
        "base_kl",
        "hidden_state_matching",
        "activation_direction",
    }
    assert {item.value for item in AssertionType} == {
        "token_log_probability",
        "sequence_log_probability",
        "sequence_margin",
        "multiple_choice_margin",
        "exact_match",
        "normalized_exact_match",
        "regular_expression",
        "json_parse",
        "json_schema",
        "free_generation_match",
        "reference_kl",
        "base_kl",
        "generation_length",
        "perplexity",
    }


def test_static_exact_output_contradiction_has_prompt_witness() -> None:
    first = loads_contract(contract_text(contract_id="one", target=None))
    second = loads_contract(
        contract_text(contract_id="two").replace(
            "expected: '{\"answer\": 3}'", "expected: '{\"answer\": 4}'"
        )
    )
    records = {
        "one": {"probes/validation.jsonl": ({"prompt": "answer"},)},
        "two": {"probes/validation.jsonl": ({"prompt": "answer"},)},
    }
    result = check_static_contracts((first, second), records_by_contract=records)
    assert result.contradictory
    assert any(item.code == "INCOMPATIBLE_EXACT_REQUIREMENTS" for item in result.witnesses)


def test_static_opposite_choice_margin_and_identity_contradictions() -> None:
    target_one = """
    - id: choose
      type: multiple_choice_margin
      source: choices.jsonl
      prompt: pick
      choices: [A, B]
      correct_choice: A
      minimum_margin: 0.1
"""
    target_two = target_one.replace("correct_choice: A", "correct_choice: B")
    first = loads_contract(contract_text(contract_id="one", target=target_one))
    second = loads_contract(
        contract_text(
            contract_id="two",
            tokenizer_hash=HASH_B,
            base_signature=HASH_B,
            target=target_two,
        )
    )
    result = check_static_contracts((first, second))
    codes = {item.code for item in result.witnesses}
    assert "INCOMPATIBLE_EXACT_REQUIREMENTS" in codes
    assert "INCOMPATIBLE_BASE_SIGNATURES" in codes
    assert "INCOMPATIBLE_TOKENIZERS" in codes


def test_no_static_contradiction_does_not_claim_satisfiability() -> None:
    result = check_static_contracts((loads_contract(contract_text()),))
    assert not result.contradictory
    assert "not established" in result.conclusion
