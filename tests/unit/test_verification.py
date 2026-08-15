from __future__ import annotations

from pathlib import Path

import pytest

from modelpact.contracts import EvaluationRecord, HoldoutPhase, SealedHoldoutGate, loads_contract
from modelpact.status import PatchClaim, VerificationOutcome
from modelpact.util.hashing import hash_canonical, sha256_bytes, sha256_file
from modelpact.verify import (
    CertificateExpectations,
    CertificateIntegrityError,
    ExecutionIdentity,
    GeneratedOutput,
    GenerationRequest,
    MappingRecordProvider,
    build_certificate,
    independently_verify,
    validate_certificate,
    verify_contract,
)
from modelpact.verify.certificate import certificate_from_dict
from modelpact.verify.generation import record_generated_output

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def verification_contract() -> object:
    return loads_contract(
        f"""
schema_version: 1
id: verify-test
model_requirements:
  tokenizer_hash: {HASH_A}
  base_signature: base-v1
  adapter_id: tiny
  output_semantics: causal_lm
compile:
  objectives:
    - {{id: teacher, type: teacher_cross_entropy, source: train.jsonl}}
verify:
  targets:
    - {{id: target, type: exact_match, source: targets.jsonl, expected: okay}}
  guards:
    - {{id: guard, type: normalized_exact_match, source: guards.jsonl, expected: base}}
holdout:
  sealed: true
  targets: holdout-targets.jsonl
  guards: holdout-guards.jsonl
  unseal_policy: final_candidate_only
statistics: {{bootstrap_samples: 10, bootstrap_seed: 3}}
generation: {{mode: greedy, max_new_tokens: 4}}
"""
    )


def identity(*, tokenizer_hash: str = HASH_A) -> ExecutionIdentity:
    return ExecutionIdentity(
        adapter_id="tiny",
        base_signature="base-v1",
        tokenizer_hash=tokenizer_hash,
    )


def generation_evidence(contract: object) -> tuple[object, ...]:
    request = GenerationRequest("target", "prompt")
    output = GeneratedOutput("okay", (1, 2), (-0.1, -0.2))
    return (
        record_generated_output(
            request,
            output,
            policy=contract.generation,  # type: ignore[attr-defined]
            seed=0,
        ),
    )


def provider() -> MappingRecordProvider:
    return MappingRecordProvider(
        {
            "targets.jsonl": (EvaluationRecord("target", "p", generated_text="okay"),),
            "guards.jsonl": (EvaluationRecord("guard", "q", generated_text=" base "),),
            "holdout-targets.jsonl": (
                EvaluationRecord("holdout-target", "hp", generated_text="okay"),
            ),
            "holdout-guards.jsonl": (
                EvaluationRecord("holdout-guard", "hg", generated_text="base"),
            ),
        }
    )


def passing_report(*, holdout: bool = False) -> object:
    contract = verification_contract()
    gate = None
    capability = None
    if holdout:
        gate = SealedHoldoutGate(contract)  # type: ignore[arg-type]
        gate.select_final_candidate("patch-1")
        capability = gate.authorize(
            phase=HoldoutPhase.FINAL_CANDIDATE,
            candidate_id="patch-1",
        )
    return verify_contract(
        contract,  # type: ignore[arg-type]
        identity=identity(),
        provider=provider(),
        free_generation_records=generation_evidence(contract),  # type: ignore[arg-type]
        probe_hashes={"targets.jsonl": HASH_B},
        include_holdout=holdout,
        holdout_gate=gate,
        holdout_capability=capability,
    )


def test_verification_executes_targets_guards_holdout_and_generation() -> None:
    report = passing_report(holdout=True)
    assert report.outcome is VerificationOutcome.PASS  # type: ignore[attr-defined]
    assert report.holdout_outcome is VerificationOutcome.PASS  # type: ignore[attr-defined]
    assert len(report.target_results) == len(report.guard_results) == 1  # type: ignore[attr-defined]
    assert len(report.holdout_target_results) == len(report.holdout_guard_results) == 1  # type: ignore[attr-defined]
    assert not report.prompt_failures  # type: ignore[attr-defined]
    assert report.result_hash.startswith("sha256:")  # type: ignore[attr-defined]


def test_failed_sealed_holdout_makes_overall_verification_fail() -> None:
    contract = verification_contract()
    failing_provider = MappingRecordProvider(
        {
            "targets.jsonl": (EvaluationRecord("target", "p", generated_text="okay"),),
            "guards.jsonl": (EvaluationRecord("guard", "q", generated_text="base"),),
            "holdout-targets.jsonl": (
                EvaluationRecord("holdout-target", "hp", generated_text="wrong"),
            ),
            "holdout-guards.jsonl": (
                EvaluationRecord("holdout-guard", "hg", generated_text="base"),
            ),
        }
    )
    gate = SealedHoldoutGate(contract)
    gate.select_final_candidate("patch-holdout-failure")
    capability = gate.authorize(
        phase=HoldoutPhase.FINAL_CANDIDATE,
        candidate_id="patch-holdout-failure",
    )

    report = verify_contract(
        contract,
        identity=identity(),
        provider=failing_provider,
        free_generation_records=generation_evidence(contract),
        include_holdout=True,
        holdout_gate=gate,
        holdout_capability=capability,
    )

    assert report.holdout_outcome is VerificationOutcome.FAIL
    assert report.outcome is VerificationOutcome.FAIL
    assert "SEALED_HOLDOUT_VERIFIED" in report.unsupported_claims


def test_generative_success_without_generation_evidence_is_inconclusive() -> None:
    contract = verification_contract()
    report = verify_contract(
        contract,  # type: ignore[arg-type]
        identity=identity(),
        provider=provider(),
    )
    assert report.outcome is VerificationOutcome.INCONCLUSIVE
    assert "FREE_GENERATION_VERIFIED" in report.unsupported_claims


def test_target_only_contract_can_be_scoped_but_not_successfully_certified() -> None:
    contract = loads_contract(
        f"""
schema_version: 1
id: target-only
model_requirements: {{tokenizer_hash: {HASH_A}, output_semantics: causal_lm}}
compile: {{objectives: []}}
verify:
  targets:
    - {{id: target, type: exact_match, source: targets.jsonl, expected: okay}}
  guards: []
holdout: {{sealed: true}}
statistics: {{bootstrap_samples: 10, bootstrap_seed: 3}}
generation: {{mode: greedy, max_new_tokens: 1}}
"""
    )
    report = verify_contract(
        contract,
        identity=identity(),
        provider=MappingRecordProvider(
            {"targets.jsonl": (EvaluationRecord("target", "p", generated_text="okay"),)}
        ),
        free_generation_records=generation_evidence(contract),
    )
    assert report.outcome is VerificationOutcome.PASS
    assert "PRESERVATION_ASSERTIONS_VERIFIED" in report.unsupported_claims
    with pytest.raises(CertificateIntegrityError, match="preservation guard"):
        build_certificate(
            report,
            contract,
            patch_id="target-only-patch",
            checkpoint_hashes={"base": HASH_A},
            artifact_hashes={},
        )


def test_identity_mismatch_is_a_failure_not_compatibility_success() -> None:
    contract = verification_contract()
    report = verify_contract(
        contract,  # type: ignore[arg-type]
        identity=identity(tokenizer_hash=HASH_B),
        provider=provider(),
        free_generation_records=generation_evidence(contract),  # type: ignore[arg-type]
    )
    assert report.outcome is VerificationOutcome.FAIL
    assert any("tokenizer_hash mismatch" in item for item in report.compatibility_errors)


def test_certificate_is_content_addressed_and_claims_match_evidence(tmp_path: Path) -> None:
    contract = verification_contract()
    report = passing_report(holdout=True)
    artifact = tmp_path / "delta-program.json"
    artifact.write_text("{}")
    certificate = build_certificate(
        report,  # type: ignore[arg-type]
        contract,  # type: ignore[arg-type]
        patch_id="patch-1",
        checkpoint_hashes={"model.safetensors": HASH_A},
        artifact_hashes={"delta-program.json": sha256_file(artifact)},
        objectives_optimized=True,
    )
    assert PatchClaim.TARGET_ASSERTIONS_VERIFIED.value in certificate.claims
    assert PatchClaim.PRESERVATION_ASSERTIONS_VERIFIED.value in certificate.claims
    assert PatchClaim.SEALED_HOLDOUT_VERIFIED.value in certificate.claims
    assert PatchClaim.FREE_GENERATION_VERIFIED.value in certificate.claims
    validate_certificate(
        certificate,
        expectations=CertificateExpectations(
            patch_id="patch-1",
            base_signature="base-v1",
            tokenizer_hash=HASH_A,
            contract_hashes={contract.id: contract.contract_id},  # type: ignore[attr-defined]
            verification_result_hash=report.result_hash,  # type: ignore[attr-defined]
        ),
        artifact_root=tmp_path,
    )


def test_certificate_result_mutation_is_detected() -> None:
    contract = verification_contract()
    report = passing_report()
    certificate = build_certificate(
        report,  # type: ignore[arg-type]
        contract,  # type: ignore[arg-type]
        patch_id="patch-1",
        checkpoint_hashes={"model": HASH_A},
        artifact_hashes={},
    )
    value = certificate.to_dict()
    targets = value["target_assertions"]
    assert isinstance(targets, list)
    targets[0]["outcome"] = "FAIL"
    with pytest.raises(CertificateIntegrityError, match="content hash mismatch"):
        certificate_from_dict(value)


def test_even_rehashed_unsupported_claim_is_rejected() -> None:
    contract = verification_contract()
    certificate = build_certificate(
        passing_report(),  # type: ignore[arg-type]
        contract,  # type: ignore[arg-type]
        patch_id="patch-1",
        checkpoint_hashes={"model": HASH_A},
        artifact_hashes={},
    )
    value = certificate.to_dict()
    value["claims"] = [*value["claims"], "UNIVERSALLY_SAFE"]
    payload = dict(value)
    del payload["certificate_hash"]
    value["certificate_hash"] = hash_canonical(payload)
    with pytest.raises(ValueError, match="unknown claim"):
        certificate_from_dict(value)


def test_certificate_detects_artifact_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "tensor.bin"
    artifact.write_bytes(b"original")
    contract = verification_contract()
    certificate = build_certificate(
        passing_report(),  # type: ignore[arg-type]
        contract,  # type: ignore[arg-type]
        patch_id="patch-1",
        checkpoint_hashes={"model": HASH_A},
        artifact_hashes={"tensor.bin": sha256_file(artifact)},
    )
    artifact.write_bytes(b"changed")
    with pytest.raises(CertificateIntegrityError, match="artifact hash mismatch"):
        validate_certificate(certificate, artifact_root=tmp_path)


def test_independent_verification_rehashes_and_reexecutes(tmp_path: Path) -> None:
    artifact = tmp_path / "patch.json"
    artifact.write_text("{}")
    contract = verification_contract()
    result = independently_verify(
        contract,  # type: ignore[arg-type]
        patch_id="patch-independent",
        identity=identity(),
        provider=provider(),
        checkpoint_hashes={"base": HASH_A},
        artifact_root=tmp_path,
        artifact_paths=("patch.json",),
        probe_hashes={"targets.jsonl": sha256_bytes(b"probe")},
        free_generation_records=generation_evidence(contract),  # type: ignore[arg-type]
        include_holdout=False,
    )
    assert result.report.outcome is VerificationOutcome.PASS
    assert result.certificate.patch_id == "patch-independent"
    assert any("independent execution" in item for item in result.certificate.warnings)
