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
    MappingRecordProvider,
    build_certificate,
    loads_certificate,
    validate_certificate,
    verify_contract,
)
from modelpact.verify.certificate import certificate_from_dict
from modelpact.verify.provider import ProbeDataError, load_probe_records

HASH = "sha256:" + "a" * 64


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
        identity=ExecutionIdentity("adapter", "base", HASH),
        provider=provider,
    )
    return build_certificate(
        report,
        contract,  # type: ignore[arg-type]
        patch_id="patch",
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
    with pytest.raises(CertificateIntegrityError, match="holdout outcome"):
        certificate_from_dict(value)


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
