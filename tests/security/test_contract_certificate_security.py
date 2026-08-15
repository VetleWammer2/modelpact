from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelpact.contracts import (
    AssertionType,
    ContractResourceLimitError,
    ContractSyntaxError,
    ContractValidationError,
    EvaluationRecord,
    VerificationAssertion,
    evaluate_assertion,
    loads_contract,
)
from modelpact.status import VerificationOutcome
from modelpact.util.hashing import hash_canonical, sha256_file
from modelpact.verify import (
    CertificateError,
    CertificateIntegrityError,
    ExecutionIdentity,
    GeneratedOutput,
    GenerationRequest,
    MappingRecordProvider,
    build_certificate,
    loads_certificate,
    validate_certificate,
    verify_contract,
)
from modelpact.verify.certificate import certificate_from_dict
from modelpact.verify.generation import record_generated_output
from modelpact.verify.provider import ProbeDataError, load_probe_records

HASH = "sha256:" + "a" * 64
BASE_SIGNATURE = "sha256:" + "b" * 64
PATCH_ID = "sha256:" + "c" * 64
GENERATED_PATCH_ID = "sha256:" + "d" * 64


def minimal_contract() -> object:
    return loads_contract(
        f"""
schema_version: 1
id: secure
model_requirements: {{tokenizer_hash: {HASH}, output_semantics: causal_lm}}
compile: {{objectives: []}}
verify:
  targets:
    - {{id: score, type: token_log_probability, source: probes.jsonl, minimum: -10}}
  guards:
    - {{id: guard, type: base_kl, source: guards.jsonl, maximum_mean: 1}}
holdout: {{sealed: true}}
statistics: {{bootstrap_samples: 2, bootstrap_seed: 1}}
generation: {{mode: greedy, max_new_tokens: 1}}
"""
    )


def certificate() -> object:
    contract = minimal_contract()
    provider = MappingRecordProvider(
        {
            "probes.jsonl": (EvaluationRecord("p", "p", values={"token_log_probability": -1.0}),),
            "guards.jsonl": (EvaluationRecord("g", "g", values={"base_kl": 0.0}),),
        }
    )
    report = verify_contract(
        contract,  # type: ignore[arg-type]
        identity=ExecutionIdentity("adapter", BASE_SIGNATURE, HASH),
        provider=provider,
    )
    return build_certificate(
        report,
        contract,  # type: ignore[arg-type]
        patch_id=PATCH_ID,
        checkpoint_hashes={"base": HASH},
        artifact_hashes={},
    )


def generative_certificate() -> object:
    contract = loads_contract(
        f"""
schema_version: 1
id: generated-secure
model_requirements: {{tokenizer_hash: {HASH}, output_semantics: causal_lm}}
compile: {{objectives: []}}
verify:
  targets:
    - {{id: generated, type: exact_match, source: generated.jsonl, expected: okay}}
  guards:
    - {{id: generated-guard, type: exact_match, source: guard.jsonl, expected: base}}
holdout: {{sealed: true}}
statistics: {{bootstrap_samples: 2, bootstrap_seed: 1}}
generation: {{mode: greedy, max_new_tokens: 1}}
"""
    )
    output = GeneratedOutput("okay", (1,), (-0.1,), {"passed": True})
    generation = record_generated_output(
        GenerationRequest("generated", "prompt"),
        output,
        policy=contract.generation,
        seed=0,
    )
    report = verify_contract(
        contract,
        identity=ExecutionIdentity("adapter", BASE_SIGNATURE, HASH),
        provider=MappingRecordProvider(
            {
                "generated.jsonl": (
                    EvaluationRecord("generated", "prompt", generated_text="okay"),
                ),
                "guard.jsonl": (
                    EvaluationRecord("generated-guard", "guard", generated_text="base"),
                ),
            }
        ),
        free_generation_records=(generation,),
    )
    return build_certificate(
        report,
        contract,
        patch_id=GENERATED_PATCH_ID,
        checkpoint_hashes={"base": HASH},
        artifact_hashes={},
    )


def aggregate_generative_certificate(assertion_type: str) -> object:
    contract = loads_contract(
        f"""
schema_version: 1
id: generated-aggregate-{assertion_type}
model_requirements: {{tokenizer_hash: {HASH}, output_semantics: causal_lm}}
compile: {{objectives: []}}
verify:
  targets:
    - id: generated
      type: {assertion_type}
      source: generated.jsonl
      expected: okay
      minimum_pass_rate: 0.5
  guards:
    - {{id: guard, type: base_kl, source: guard.jsonl, maximum_mean: 1.0}}
holdout: {{sealed: true}}
statistics: {{bootstrap_samples: 2, bootstrap_seed: 1}}
generation: {{mode: greedy, max_new_tokens: 1}}
"""
    )
    passing_generation = record_generated_output(
        GenerationRequest("generated-pass", "prompt-pass"),
        GeneratedOutput("okay", (1,), parser_result={"passed": True}),
        policy=contract.generation,
        seed=0,
    )
    failing_generation = record_generated_output(
        GenerationRequest("generated-fail", "prompt-fail"),
        GeneratedOutput("wrong", (2,), parser_result={"passed": False}),
        policy=contract.generation,
        seed=0,
    )
    report = verify_contract(
        contract,
        identity=ExecutionIdentity("adapter", BASE_SIGNATURE, HASH),
        provider=MappingRecordProvider(
            {
                "generated.jsonl": (
                    EvaluationRecord("generated-pass", "prompt-pass", generated_text="okay"),
                    EvaluationRecord("generated-fail", "prompt-fail", generated_text="wrong"),
                ),
                "guard.jsonl": (EvaluationRecord("guard", "guard", values={"base_kl": 0.0}),),
            }
        ),
        free_generation_records=(passing_generation, failing_generation),
    )
    return build_certificate(
        report,
        contract,
        patch_id=hash_canonical({"generated_aggregate": assertion_type}),
        checkpoint_hashes={"base": HASH},
        artifact_hashes={},
    )


def test_yaml_python_objects_and_alias_bombs_are_rejected_as_data() -> None:
    with pytest.raises(ContractSyntaxError):
        loads_contract("!!python/object/apply:os.system ['echo unsafe']")
    with pytest.raises(ContractSyntaxError, match="anchors"):
        loads_contract("a: &a [x, x, x]\nb: [*a, *a, *a]\n")


def test_regex_features_with_unbounded_backtracking_are_rejected() -> None:
    text = """
schema_version: 1
id: regex
model_requirements: {output_semantics: causal_lm}
compile: {objectives: []}
verify:
  targets:
    - id: unsafe
      type: regular_expression
      source: probes.jsonl
      pattern: '(a+)+$'
  guards: []
holdout: {sealed: true}
statistics: {bootstrap_samples: 2}
generation: {mode: greedy, max_new_tokens: 1}
"""
    with pytest.raises(ContractValidationError, match="quantified groups"):
        loads_contract(text)


def test_excessively_nested_contract_is_rejected_before_ast_construction() -> None:
    value: object = "leaf"
    for _ in range(80):
        value = {"x": value}
    with pytest.raises(ContractResourceLimitError, match="nesting depth"):
        loads_contract(json.dumps(value), format="json")


def test_duplicate_generated_json_keys_fail_json_parse_assertion() -> None:
    result = evaluate_assertion(
        VerificationAssertion("json", AssertionType.JSON_PARSE, "p.jsonl"),
        (EvaluationRecord("p", "prompt", generated_text='{"a":1,"a":2}'),),
    )
    assert result.outcome is VerificationOutcome.FAIL


def test_probe_loader_rejects_duplicate_keys_precomputed_results_and_traversal(
    tmp_path: Path,
) -> None:
    (tmp_path / "duplicate.jsonl").write_text('{"prompt":"x","prompt":"y"}\n')
    with pytest.raises(ContractSyntaxError, match="duplicate JSON"):
        load_probe_records(tmp_path, "duplicate.jsonl")
    (tmp_path / "spoof.jsonl").write_text('{"prompt":"x","base_kl":0}\n')
    with pytest.raises(ProbeDataError, match="unknown probe field"):
        load_probe_records(tmp_path, "spoof.jsonl")
    with pytest.raises(ValueError, match="unsafe relative path"):
        load_probe_records(tmp_path, "../outside.jsonl")


def test_certificate_duplicate_keys_and_unknown_fields_are_rejected() -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    text = json.dumps(value)
    duplicate = text[:-1] + ',"schema_version":1}'
    with pytest.raises(ContractSyntaxError, match="duplicate JSON"):
        loads_certificate(duplicate)
    value["arbitrary_python"] = "module:function"
    payload = dict(value)
    payload.pop("certificate_hash")
    value["certificate_hash"] = hash_canonical(payload)
    with pytest.raises(CertificateError, match="unknown certificate field"):
        certificate_from_dict(value)


def test_certificate_artifact_paths_cannot_escape_even_with_valid_self_hash() -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    value["artifact_hashes"] = {"../secret": HASH}
    payload = dict(value)
    payload.pop("certificate_hash")
    value["certificate_hash"] = hash_canonical(payload)
    with pytest.raises(ValueError, match="unsafe relative path"):
        certificate_from_dict(value)


def test_claim_insertion_is_rejected_after_attacker_recomputes_self_hash() -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    value["claims"] = [*value["claims"], "SEALED_HOLDOUT_VERIFIED"]
    value["unsupported_claims"] = [
        item for item in value["unsupported_claims"] if item != "SEALED_HOLDOUT_VERIFIED"
    ]
    payload = dict(value)
    payload.pop("certificate_hash")
    value["certificate_hash"] = hash_canonical(payload)
    with pytest.raises(CertificateIntegrityError, match="holdout"):
        certificate_from_dict(value)


def _rehash(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("certificate_hash")
    value["certificate_hash"] = hash_canonical(payload)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("patch_id", "patch"),
        ("patch_id", "sha256:" + "A" * 64),
        ("base_signature", "base"),
        ("base_signature", "sha256:" + "b" * 63),
    ],
)
def test_rehashed_certificate_rejects_malformed_core_identity_digest(
    field: str,
    malformed: str,
) -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    value[field] = malformed
    _rehash(value)

    with pytest.raises(CertificateError, match=rf"{field} must be a lowercase sha256"):
        certificate_from_dict(value)


@pytest.mark.parametrize("field", ["patch_id", "base_signature", "model_adapter_id"])
def test_rehashed_certificate_rejects_missing_core_identity_field(field: str) -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    del value[field]
    _rehash(value)

    with pytest.raises(CertificateError, match=rf"missing certificate field.*{field}"):
        certificate_from_dict(value)


def test_rehashed_guard_failure_cannot_remain_an_overall_pass() -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    guards = value["guard_assertions"]
    assert isinstance(guards, list)
    guards[0]["outcome"] = "FAIL"
    value["claims"] = [
        item for item in value["claims"] if item != "PRESERVATION_ASSERTIONS_VERIFIED"
    ]
    _rehash(value)
    with pytest.raises(CertificateIntegrityError, match="non-passing assertion"):
        certificate_from_dict(value)


def test_rehashed_holdout_failure_cannot_remain_an_overall_pass() -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    targets = value["target_assertions"]
    assert isinstance(targets, list)
    value["sealed_holdout_result"] = {
        "outcome": "FAIL",
        "targets": [targets[0]],
        "guards": [],
    }
    _rehash(value)
    with pytest.raises(CertificateIntegrityError, match="sealed holdout"):
        certificate_from_dict(value)


def test_rehashed_failed_generation_parser_cannot_retain_pass() -> None:
    value = generative_certificate().to_dict()  # type: ignore[union-attr]
    generation = value["free_generation_results"]
    assert isinstance(generation, list)
    generation[0]["parser_result"]["passed"] = False
    value["claims"] = [item for item in value["claims"] if item != "FREE_GENERATION_VERIFIED"]
    _rehash(value)
    with pytest.raises(CertificateIntegrityError, match="free-generation"):
        certificate_from_dict(value)


@pytest.mark.parametrize("assertion_type", ["exact_match", "free_generation_match"])
def test_certificate_accepts_aggregate_generation_pass_with_one_prompt_failure(
    assertion_type: str,
) -> None:
    cert = aggregate_generative_certificate(assertion_type)
    value = cert.to_dict()  # type: ignore[union-attr]
    target = value["target_assertions"][0]
    assert target["outcome"] == "PASS"
    assert target["value"] == 0.5
    assert [item["outcome"] for item in target["prompt_metrics"]] == ["PASS", "FAIL"]
    validate_certificate(cert)  # type: ignore[arg-type]


def test_rehashed_aggregate_generation_tampering_is_rejected() -> None:
    value = aggregate_generative_certificate("free_generation_match").to_dict()  # type: ignore[union-attr]
    generation = value["free_generation_results"]
    assert isinstance(generation, list)
    generation[0]["parser_result"]["passed"] = False
    _rehash(value)
    with pytest.raises(CertificateIntegrityError, match="free-generation"):
        certificate_from_dict(value)

    value = aggregate_generative_certificate("exact_match").to_dict()  # type: ignore[union-attr]
    targets = value["target_assertions"]
    assert isinstance(targets, list)
    targets[0]["prompt_metrics"][0]["outcome"] = "FAIL"
    _rehash(value)
    with pytest.raises(CertificateIntegrityError, match="aggregate evidence"):
        certificate_from_dict(value)


def test_rehashed_continuous_result_cannot_retain_a_passing_margin() -> None:
    value = certificate().to_dict()  # type: ignore[union-attr]
    guards = value["guard_assertions"]
    prompt_metrics = value["prompt_level_metrics"]
    assert isinstance(guards, list)
    assert isinstance(prompt_metrics, list)
    guards[0]["value"] = 999.0
    guards[0]["prompt_metrics"][0]["value"] = 999.0
    for metric in prompt_metrics:
        if metric.get("role") == "guard":
            metric["value"] = 999.0
    _rehash(value)
    with pytest.raises(CertificateIntegrityError, match="aggregate evidence"):
        certificate_from_dict(value)


def test_certificate_acceptance_binding_preserves_prefixed_assertion_ids() -> None:
    contract = loads_contract(
        f"""
schema_version: 1
id: prefixed-ids
model_requirements: {{tokenizer_hash: {HASH}, output_semantics: causal_lm}}
compile: {{objectives: []}}
verify:
  targets:
    - {{id: score, type: token_log_probability, source: a.jsonl, minimum: -2}}
    - {{id: score:detail, type: token_log_probability, source: b.jsonl, minimum: -2}}
  guards:
    - {{id: guard, type: base_kl, source: g.jsonl, maximum_mean: 1}}
holdout: {{sealed: true}}
statistics: {{bootstrap_samples: 2, bootstrap_seed: 1}}
generation: {{mode: greedy, max_new_tokens: 1}}
"""
    )
    report = verify_contract(
        contract,
        identity=ExecutionIdentity("adapter", BASE_SIGNATURE, HASH),
        provider=MappingRecordProvider(
            {
                "a.jsonl": (EvaluationRecord("a", "a", values={"token_log_probability": -1}),),
                "b.jsonl": (EvaluationRecord("b", "b", values={"token_log_probability": -1}),),
                "g.jsonl": (EvaluationRecord("g", "g", values={"base_kl": 0}),),
            }
        ),
    )
    built = build_certificate(
        report,
        contract,
        patch_id=hash_canonical({"patch": "prefixed"}),
        checkpoint_hashes={"base": HASH},
        artifact_hashes={},
    )
    validate_certificate(built)


def test_expected_certificate_identity_detects_contract_tokenizer_and_result_mutation() -> None:
    cert = certificate()
    with pytest.raises(CertificateIntegrityError, match="tokenizer_hash mismatch"):
        validate_certificate(
            cert,  # type: ignore[arg-type]
            expectations=__import__(
                "modelpact.verify", fromlist=["CertificateExpectations"]
            ).CertificateExpectations(tokenizer_hash="sha256:" + "b" * 64),
        )


def test_artifact_hash_verification_reads_data_but_executes_nothing(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    artifact = tmp_path / "payload.json"
    artifact.write_text(json.dumps({"call": f"touch {marker}"}))
    contract = minimal_contract()
    base = certificate()
    value = base.to_dict()  # type: ignore[union-attr]
    value["artifact_hashes"] = {"payload.json": sha256_file(artifact)}
    payload = dict(value)
    payload.pop("certificate_hash")
    value["certificate_hash"] = hash_canonical(payload)
    cert = certificate_from_dict(value)
    validate_certificate(cert, artifact_root=tmp_path)
    assert not marker.exists()
    assert contract.contract_id.startswith("sha256:")  # type: ignore[attr-defined]


def test_certificate_rejects_oversized_artifact_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "payload.json"
    artifact.write_bytes(b"{}")
    value = certificate().to_dict()  # type: ignore[union-attr]
    value["artifact_hashes"] = {"payload.json": sha256_file(artifact)}
    _rehash(value)
    cert = certificate_from_dict(value)
    monkeypatch.setattr("modelpact.verify.certificate._MAX_CERTIFICATE_ARTIFACT_BYTES", 1)

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized certificate artifact was hashed")

    monkeypatch.setattr("modelpact.verify.certificate.sha256_file", forbidden_hash)
    with pytest.raises(CertificateIntegrityError, match="exceeds size limit"):
        validate_certificate(cert, artifact_root=tmp_path)
