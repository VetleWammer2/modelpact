from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from modelpact.adapters.tiny_lm import TinyCausalLM, TinyConfig
from modelpact.contracts import EvaluationRecord
from modelpact.contracts.parser import parse_contract
from modelpact.models.manifest import ModelSignature
from modelpact.models.schema import inspect_state_schema
from modelpact.patch.ast import DeltaProgram, VectorDelta
from modelpact.patch.bundle import create_patch_bundle, load_patch_bundle
from modelpact.rebase.direct import RebaseCompatibility
from modelpact.rebase.evidence import (
    MAX_REBASE_EVIDENCE_BYTES,
    RebaseEvidence,
    RebaseEvidenceExpectations,
    loads_rebase_evidence,
    read_rebase_evidence,
    validate_rebase_evidence,
    write_rebase_evidence,
)
from modelpact.status import RebaseClaim, VerificationOutcome
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_file
from modelpact.verify import (
    ExecutionIdentity,
    MappingRecordProvider,
    VerificationCertificate,
    build_certificate,
    validate_certificate,
    verify_contract,
)
from modelpact.verify.certificate import certificate_from_dict


def _digest(index: int) -> str:
    return f"sha256:{index:064x}"


def _evidence(
    *,
    source_patch_id: str = _digest(1),
    source_base_hash: str = _digest(2),
    target_base_hash: str = _digest(3),
    target_contract_id: str = _digest(4),
    preservation_contract_id: str = _digest(5),
    old_contract_id: str | None = None,
) -> RebaseEvidence:
    return RebaseEvidence(
        source_patch_id=source_patch_id,
        source_base_hash=source_base_hash,
        target_base_hash=target_base_hash,
        claim=RebaseClaim.SEMANTIC_REBASE_VERIFIED,
        compatibility=RebaseCompatibility.DIRECT_PHYSICAL_TRANSFER.value,
        direct_attempted=True,
        direct_outcome=VerificationOutcome.FAIL.value,
        recompile_attempted=True,
        recompile_steps=17,
        recompile_restarts=1,
        budget_exhausted=False,
        old_patched_behavior={old_contract_id or target_contract_id: 0.75},
        new_patched_behavior={target_contract_id: 0.5},
        new_base_preservation={f"{preservation_contract_id}:guards": 0.25},
        patch_complexity_before={"parameters": 64, "target_tensors": 2},
        patch_complexity_after={"active_modules": 1, "parameters": 32, "total_rank": 1},
        warnings=(),
    )


def _record_bytes(value: dict[str, object]) -> bytes:
    return (canonical_dumps(value) + "\n").encode("utf-8")


def _rehash(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("evidence_hash", None)
    value["evidence_hash"] = hash_canonical(payload)


def _rehash_certificate(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("certificate_hash", None)
    value["certificate_hash"] = hash_canonical(payload)


def _rebase_certificate(evidence: RebaseEvidence) -> VerificationCertificate:
    tokenizer_hash = _digest(401)
    contract = parse_contract(
        {
            "compile": {"objectives": []},
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": "rebase-certificate-security",
            "model_requirements": {
                "output_semantics": "causal_lm",
                "tokenizer_hash": tokenizer_hash,
            },
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 2,
                "bootstrap_seed": 1,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [
                    {
                        "id": "guard",
                        "maximum_mean": 1.0,
                        "source": "guards.jsonl",
                        "type": "base_kl",
                    }
                ],
                "targets": [
                    {
                        "id": "score",
                        "minimum": -10.0,
                        "source": "probes.jsonl",
                        "type": "token_log_probability",
                    }
                ],
            },
        }
    )
    evidence = replace(
        evidence,
        new_patched_behavior={contract.contract_id: 0.5},
        new_base_preservation={f"{contract.contract_id}:guards": 0.25},
    )
    report = verify_contract(
        contract,
        identity=ExecutionIdentity(
            "test.rebase.security",
            evidence.target_base_hash,
            tokenizer_hash,
        ),
        provider=MappingRecordProvider(
            {
                "guards.jsonl": (EvaluationRecord("guard", "guard", values={"base_kl": 0.0}),),
                "probes.jsonl": (
                    EvaluationRecord(
                        "probe",
                        "probe",
                        values={"token_log_probability": -1.0},
                    ),
                ),
            }
        ),
    )
    guard_reference = next(iter(evidence.new_base_preservation))
    return build_certificate(
        report,
        contract,
        patch_id=_digest(402),
        checkpoint_hashes={"base": _digest(403)},
        artifact_hashes={
            "evidence/rebase.json": _digest(404),
            "evidence/source-manifest.json": _digest(405),
        },
        rebase_result={
            "claim": evidence.claim.value,
            "evidence": evidence.to_dict(),
            "new_base_guard_ids": [guard_reference],
            "source_base_hash": evidence.source_base_hash,
            "source_patch_id": evidence.source_patch_id,
            "target_base_hash": evidence.target_base_hash,
        },
    )


def test_rebase_evidence_canonical_round_trip_and_file_reader(tmp_path: Path) -> None:
    expected = _evidence()
    value = expected.to_dict()
    encoded = _record_bytes(value)

    assert loads_rebase_evidence(encoded).to_dict() == value
    path = tmp_path / "rebase-evidence.json"
    path.write_bytes(encoded)
    assert read_rebase_evidence(path).to_dict() == value


def test_rebase_evidence_changed_hash_with_valid_payload_is_rejected() -> None:
    value = _evidence().to_dict()
    replacement = _digest(99)
    assert replacement != value["evidence_hash"]
    value["evidence_hash"] = replacement

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


@pytest.mark.parametrize(
    "mutation",
    [
        "direct-pass-before-recompile",
        "verified-incompatible-semantics",
        "verified-without-recompile",
        "verified-budget-exhausted",
        "verified-negative-target-margin",
        "verified-erased-direct-attempt-history",
        "direct-outcome-without-attempt",
        "direct-claim-retains-recompile",
    ],
)
def test_fully_rehashed_semantic_attacks_are_rejected(mutation: str) -> None:
    value = _evidence().to_dict()
    if mutation == "direct-pass-before-recompile":
        value["direct_outcome"] = VerificationOutcome.PASS.value
    elif mutation == "verified-incompatible-semantics":
        value["compatibility"] = RebaseCompatibility.INCOMPATIBLE_TOKENIZER.value
        value["direct_attempted"] = False
        value["direct_outcome"] = None
    elif mutation == "verified-without-recompile":
        value["recompile_attempted"] = False
        value["recompile_steps"] = 0
        value["recompile_restarts"] = 0
    elif mutation == "verified-budget-exhausted":
        value["budget_exhausted"] = True
    elif mutation == "verified-negative-target-margin":
        margins = value["new_patched_behavior"]
        assert isinstance(margins, dict)
        margins[next(iter(margins))] = -0.01
    elif mutation == "verified-erased-direct-attempt-history":
        value["direct_attempted"] = False
        value["direct_outcome"] = None
    elif mutation == "direct-outcome-without-attempt":
        value["direct_attempted"] = False
    elif mutation == "direct-claim-retains-recompile":
        value["claim"] = RebaseClaim.DIRECT_TRANSPLANT_VERIFIED.value
        value["direct_outcome"] = VerificationOutcome.PASS.value
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(mutation)
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


def test_fully_rehashed_direct_claim_rejects_negative_old_behavior() -> None:
    value = _evidence().to_dict()
    value.update(
        {
            "claim": RebaseClaim.DIRECT_TRANSPLANT_VERIFIED.value,
            "direct_outcome": VerificationOutcome.PASS.value,
            "old_patched_behavior": {_digest(4): -0.01},
            "patch_complexity_after": copy.deepcopy(value["patch_complexity_before"]),
            "recompile_attempted": False,
            "recompile_restarts": 0,
            "recompile_steps": 0,
        }
    )
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


def test_rebase_evidence_duplicate_json_keys_are_rejected() -> None:
    text = _record_bytes(_evidence().to_dict()).decode("utf-8").removesuffix("\n")
    duplicate = (text[:-1] + ',"schema_version":1}\n').encode("utf-8")

    with pytest.raises(ValueError):
        loads_rebase_evidence(duplicate)


def test_rebase_evidence_unknown_field_is_rejected_after_rehash() -> None:
    value = _evidence().to_dict()
    value["unexpected"] = {"trusted": False}
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


@pytest.mark.parametrize(
    "field",
    ["source_patch_id", "source_base_hash", "target_base_hash", "claim"],
)
def test_rebase_evidence_missing_required_identity_is_rejected_after_rehash(field: str) -> None:
    value = _evidence().to_dict()
    del value[field]
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


@pytest.mark.parametrize("schema_version", [True, 1.0, 2])
def test_rebase_evidence_rejects_ambiguous_or_unsupported_schema_versions(
    schema_version: bool | float | int,
) -> None:
    value = _evidence().to_dict()
    value["schema_version"] = schema_version
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


def test_rebase_evidence_rejects_noncanonical_representations() -> None:
    value = _evidence().to_dict()
    canonical = _record_bytes(value)
    reversed_value = dict(reversed(tuple(value.items())))
    variants = (
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        (json.dumps(reversed_value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        canonical[:-1] + b" \n",
        canonical + b"\n",
        b"\xef\xbb\xbf" + canonical,
    )

    for encoded in variants:
        with pytest.raises(ValueError):
            loads_rebase_evidence(encoded)


def test_rebase_evidence_rejects_negative_zero_instead_of_normalizing_it() -> None:
    value = _evidence().to_dict()
    margins = value["new_patched_behavior"]
    assert isinstance(margins, dict)
    margins[next(iter(margins))] = -0.0
    _rehash(value)
    # json.dumps preserves the hostile spelling; hash_canonical above normalized it.
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    assert b"-0.0" in encoded

    with pytest.raises(ValueError):
        loads_rebase_evidence(encoded)


@pytest.mark.parametrize(
    "suffix",
    [
        b"",
        b"{}\n",
        b"garbage",
    ],
)
def test_rebase_evidence_rejects_truncated_or_trailing_content(suffix: bytes) -> None:
    encoded = _record_bytes(_evidence().to_dict())
    hostile = encoded[:-2] + b"\n" if not suffix else encoded + suffix

    with pytest.raises(ValueError):
        loads_rebase_evidence(hostile)


def test_rebase_evidence_file_size_is_bounded(tmp_path: Path) -> None:
    value = _evidence().to_dict()
    value["warnings"] = ["x" * (17 * 1024 * 1024)]
    _rehash(value)
    path = tmp_path / "oversized-rebase-evidence.json"
    path.write_bytes(_record_bytes(value))

    with pytest.raises(ValueError):
        read_rebase_evidence(path)


def test_rebase_evidence_writer_rejects_oversized_record_without_output(
    tmp_path: Path,
) -> None:
    warnings = tuple(f"{index:04d}:" + "x" * 4091 for index in range(4097))
    evidence = replace(_evidence(), warnings=warnings)
    assert len(_record_bytes(evidence.to_dict())) > MAX_REBASE_EVIDENCE_BYTES
    path = tmp_path / "oversized-rebase-evidence.json"

    with pytest.raises(ValueError):
        write_rebase_evidence(evidence, path)

    assert not path.exists()


def test_rebase_evidence_nesting_depth_is_bounded() -> None:
    value = _evidence().to_dict()
    nested: object = 0
    for _ in range(80):
        nested = [nested]
    value["patch_complexity_after"] = {"nested": nested}
    # The hostile depth exceeds canonical_dumps' own output limit, so use the
    # standard encoder and let the bounded reader reject it before hash checks.
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()

    with pytest.raises(ValueError):
        loads_rebase_evidence(encoded)


def test_rebase_evidence_collection_count_is_bounded_after_rehash() -> None:
    value = _evidence().to_dict()
    value["warnings"] = [f"warning-{index}" for index in range(100_001)]
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


def test_rebase_evidence_object_count_is_bounded_after_rehash() -> None:
    value = _evidence().to_dict()
    value["new_patched_behavior"] = {f"contract-{index:06d}": 0.5 for index in range(100_001)}
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("source_patch_id", "patch"),
        ("source_patch_id", "sha256:" + "A" * 64),
        ("source_base_hash", "sha256:" + "b" * 63),
        ("target_base_hash", "sha512:" + "c" * 64),
        ("evidence_hash", "sha256:" + "D" * 64),
    ],
)
def test_rehashed_malformed_core_identities_are_rejected(field: str, malformed: str) -> None:
    value = _evidence().to_dict()
    value[field] = malformed
    if field != "evidence_hash":
        _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


@pytest.mark.parametrize(
    ("field", "reference"),
    [
        ("old_patched_behavior", "a/../source"),
        ("new_patched_behavior", "../substituted-contract"),
        ("new_base_preservation", "C:/ambiguous"),
        ("new_base_preservation", "\x00truncated"),
        ("new_patched_behavior", "x" * 129),
    ],
)
def test_rehashed_malformed_contract_reference_is_rejected(
    field: str,
    reference: str,
) -> None:
    value = _evidence().to_dict()
    value[field] = {reference: 0.5}
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


@pytest.mark.parametrize(
    ("field", "hostile"),
    [
        ("recompile_steps", True),
        ("recompile_steps", -1),
        ("recompile_steps", 2**31),
        ("recompile_restarts", True),
        ("recompile_restarts", -1),
        ("recompile_restarts", 2**31),
    ],
)
def test_rehashed_invalid_or_excessive_integers_are_rejected(
    field: str,
    hostile: bool | int,
) -> None:
    value = _evidence().to_dict()
    value[field] = hostile
    _rehash(value)

    with pytest.raises(ValueError):
        loads_rebase_evidence(_record_bytes(value))


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_rebase_evidence_nonfinite_numbers_are_rejected(nonfinite: float) -> None:
    value = _evidence().to_dict()
    value["new_patched_behavior"] = {_digest(4): nonfinite}
    encoded = (
        json.dumps(value, ensure_ascii=False, allow_nan=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()

    with pytest.raises(ValueError):
        loads_rebase_evidence(encoded)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("evidence_hash", _digest(101)),
        ("source_patch_id", _digest(102)),
        ("source_base_hash", _digest(103)),
        ("target_base_hash", _digest(104)),
        ("claim", RebaseClaim.DIRECT_TRANSPLANT_VERIFIED),
    ],
)
def test_caller_expectations_reject_validly_rehashed_identity_substitution(
    field: str,
    replacement: str | RebaseClaim,
) -> None:
    evidence = _evidence()
    value = evidence.to_dict()
    expectation_values: dict[str, Any] = {
        "evidence_hash": value["evidence_hash"],
        "source_patch_id": value["source_patch_id"],
        "source_base_hash": value["source_base_hash"],
        "target_base_hash": value["target_base_hash"],
        "claim": evidence.claim,
    }
    validate_rebase_evidence(
        evidence,
        expectations=RebaseEvidenceExpectations(**expectation_values),
    )
    expectation_values[field] = replacement

    with pytest.raises(ValueError):
        validate_rebase_evidence(
            evidence,
            expectations=RebaseEvidenceExpectations(**expectation_values),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "nested-source-patch",
        "outer-source-patch",
        "retarget-base",
        "outer-claim",
    ],
)
def test_certificate_rejects_fully_rehashed_nested_rebase_identity_substitution(
    mutation: str,
) -> None:
    value = _rebase_certificate(_evidence()).to_dict()
    rebase_result = value["rebase_result"]
    assert isinstance(rebase_result, dict)
    nested_evidence = rebase_result["evidence"]
    assert isinstance(nested_evidence, dict)

    if mutation == "nested-source-patch":
        nested_evidence["source_patch_id"] = _digest(410)
        _rehash(nested_evidence)
    elif mutation == "outer-source-patch":
        rebase_result["source_patch_id"] = _digest(411)
    elif mutation == "retarget-base":
        replacement = _digest(412)
        nested_evidence["target_base_hash"] = replacement
        rebase_result["target_base_hash"] = replacement
        _rehash(nested_evidence)
    elif mutation == "outer-claim":
        rebase_result["claim"] = RebaseClaim.DIRECT_TRANSPLANT_VERIFIED.value
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(mutation)
    _rehash_certificate(value)

    with pytest.raises(ValueError):
        certificate_from_dict(value)


@pytest.mark.parametrize("mutation", ["outer-guard", "nested-preservation"])
def test_certificate_rejects_rehashed_guard_not_bound_to_preservation_evidence(
    mutation: str,
) -> None:
    value = _rebase_certificate(_evidence()).to_dict()
    rebase_result = value["rebase_result"]
    assert isinstance(rebase_result, dict)
    if mutation == "outer-guard":
        rebase_result["new_base_guard_ids"] = [f"{_digest(420)}:guards"]
    elif mutation == "nested-preservation":
        nested_evidence = rebase_result["evidence"]
        assert isinstance(nested_evidence, dict)
        nested_evidence["new_base_preservation"] = {f"{_digest(421)}:guards": 0.25}
        _rehash(nested_evidence)
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(mutation)
    _rehash_certificate(value)

    with pytest.raises(ValueError):
        certificate_from_dict(value)


def test_certificate_rejects_rehashed_top_level_rebase_claim_without_result() -> None:
    value = _rebase_certificate(_evidence()).to_dict()
    value["rebase_result"] = {"outcome": VerificationOutcome.NOT_APPLICABLE.value}
    artifact_hashes = value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    del artifact_hashes["evidence/rebase.json"]
    del artifact_hashes["evidence/source-manifest.json"]
    claims = value["claims"]
    assert isinstance(claims, list)
    claims.append(RebaseClaim.SEMANTIC_REBASE_VERIFIED.value)
    _rehash_certificate(value)

    with pytest.raises(ValueError):
        certificate_from_dict(value)


def test_failed_verification_still_carries_its_rebase_lineage() -> None:
    """A rebased bundle that fails re-verification stays representable.

    rebase_result is packaging-time lineage, so binding it to PASS would make a
    FAIL certificate impossible to emit for a rebased bundle. The execution
    verdict is verification_outcome, and RebaseClaim values remain inadmissible
    in claims, so the record asserts nothing about this run.
    """

    value = _rebase_certificate(_evidence()).to_dict()
    value["verification_outcome"] = VerificationOutcome.FAIL.value
    _rehash_certificate(value)

    parsed = certificate_from_dict(value)

    assert parsed.verification_outcome is VerificationOutcome.FAIL
    assert parsed.rebase_result["claim"] == RebaseClaim.SEMANTIC_REBASE_VERIFIED.value
    assert not {item.value for item in RebaseClaim} & set(parsed.claims)


@pytest.mark.parametrize(
    "removed",
    ["evidence/rebase.json", "evidence/source-manifest.json"],
)
def test_certificate_rejects_rehashed_rebase_result_without_pinned_artifacts(
    removed: str,
) -> None:
    value = _rebase_certificate(_evidence()).to_dict()
    artifact_hashes = value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    del artifact_hashes[removed]
    _rehash_certificate(value)

    with pytest.raises(ValueError):
        certificate_from_dict(value)


def test_recertification_may_pin_lineage_artifacts_without_asserting_a_rebase() -> None:
    """Independent re-execution keeps the artifacts but re-derives no rebase.

    Requiring rebase_result whenever the lineage artifacts are present would
    make a rebased bundle impossible to re-certificate; independently_verify
    reports the absent record as a prior-certificate difference instead.
    """

    value = _rebase_certificate(_evidence()).to_dict()
    value["rebase_result"] = {"outcome": VerificationOutcome.NOT_APPLICABLE.value}
    _rehash_certificate(value)

    parsed = certificate_from_dict(value)

    assert parsed.rebase_result == {"outcome": VerificationOutcome.NOT_APPLICABLE.value}
    assert "evidence/rebase.json" in parsed.artifact_hashes


@pytest.mark.parametrize(
    "alias",
    [
        r"evidence\rebase.json",
        "evidence/Rebase.json",
        "evidence/rebase.json.",
        "evidence/rebase.json ",
        "evidence/rebase.json:stream",
    ],
)
def test_certificate_rejects_fully_rehashed_noncanonical_artifact_path_alias(
    alias: str,
) -> None:
    value = _rebase_certificate(_evidence()).to_dict()
    artifact_hashes = value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes[alias] = artifact_hashes["evidence/rebase.json"]
    _rehash_certificate(value)

    with pytest.raises(ValueError):
        certificate_from_dict(value)


@pytest.mark.parametrize("record", ["certificate", "evidence"])
def test_float_schema_version_is_not_read_as_v1(record: str) -> None:
    """`1.0 == 1` in Python, so a float spelling must be rejected on type.

    Accepting it would leave the self-hash addressing a payload the reader
    normalizes away, so the parsed record no longer content-addresses itself.
    """

    if record == "certificate":
        value = _rebase_certificate(_evidence()).to_dict()
        value["schema_version"] = 1.0
        _rehash_certificate(value)
        with pytest.raises(ValueError, match="schema_version 1"):
            certificate_from_dict(value)
        return
    value = _evidence().to_dict()
    value["schema_version"] = 1.0
    _rehash(value)
    with pytest.raises(ValueError, match="schema version"):
        loads_rebase_evidence(_record_bytes(value))


def test_certificate_rejects_oversized_exact_rebase_artifact_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "evidence" / "rebase.json"
    evidence_path.parent.mkdir()
    evidence_path.write_bytes(b"x" * (MAX_REBASE_EVIDENCE_BYTES + 1))

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized Rebase Evidence artifact was hashed")

    monkeypatch.setattr("modelpact.verify.certificate.sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="exceeds size limit"):
        validate_certificate(_rebase_certificate(_evidence()), artifact_root=tmp_path)


@pytest.mark.parametrize("binding", ["target", "guard"])
def test_certificate_rejects_rehashed_evidence_contract_binding_substitution(
    binding: str,
) -> None:
    value = _rebase_certificate(_evidence()).to_dict()
    contract_hashes = value["contract_hashes"]
    rebase_result = value["rebase_result"]
    assert isinstance(contract_hashes, dict)
    assert isinstance(rebase_result, dict)
    nested_evidence = rebase_result["evidence"]
    assert isinstance(nested_evidence, dict)

    expected_contract_ids = set(contract_hashes.values())
    assert set(nested_evidence["new_patched_behavior"]) == expected_contract_ids
    assert set(nested_evidence["new_base_preservation"]) == {
        f"{contract_id}:guards" for contract_id in expected_contract_ids
    }

    if binding == "target":
        nested_evidence["new_patched_behavior"] = {_digest(430): 0.5}
    elif binding == "guard":
        substituted_guard = f"{_digest(431)}:guards"
        nested_evidence["new_base_preservation"] = {substituted_guard: 0.25}
        # Keep the outer guard list internally consistent so only the missing
        # certificate contract binding distinguishes the hostile record.
        rebase_result["new_base_guard_ids"] = [substituted_guard]
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(binding)
    _rehash(nested_evidence)
    _rehash_certificate(value)

    with pytest.raises(ValueError):
        certificate_from_dict(value)


def _rebase_bundle(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    state_schema = inspect_state_schema(model)
    signature = ModelSignature(
        schema_version=1,
        adapter_id="test.rebase.security",
        architecture_hash=_digest(200),
        state_schema_hash=state_schema.schema_hash,
        checkpoint_hash=_digest(201),
        tokenizer_hash=_digest(202),
        chat_template_hash=_digest(203),
        generation_config_hash=_digest(204),
    )
    contract = parse_contract(
        {
            "compile": {"objectives": []},
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": "rebase-security-contract",
            "model_requirements": {
                "adapter_id": signature.adapter_id,
                "architecture_hash": signature.architecture_hash,
                "base_signature": signature.signature_hash,
                "output_semantics": "causal_lm",
                "state_schema_hash": signature.state_schema_hash,
                "tokenizer_hash": signature.tokenizer_hash,
            },
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 2,
                "bootstrap_seed": 1,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [
                    {
                        "id": "preserve-base",
                        "maximum_mean": 1.0,
                        "source": "guards/validation.jsonl",
                        "type": "base_kl",
                    }
                ],
                "targets": [
                    {
                        "id": "generated-length",
                        "maximum": 1,
                        "minimum": 1,
                        "source": "probes/validation.jsonl",
                        "type": "generation_length",
                    }
                ],
            },
        }
    )
    source_signature = replace(signature, checkpoint_hash=_digest(206))
    source_base_hash = source_signature.signature_hash
    source_contract_value: dict[str, Any] = copy.deepcopy(contract.to_dict())
    source_model_requirements = source_contract_value["model_requirements"]
    assert isinstance(source_model_requirements, dict)
    source_model_requirements["base_signature"] = source_base_hash
    source_contract = parse_contract(source_contract_value)
    assert source_contract.contract_id != contract.contract_id
    source_bundle = create_patch_bundle(
        tmp_path / "source-patch",
        name="rebase-evidence-source",
        base_signature=source_signature.to_dict(),
        state_schema=state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("delta")}),
        tensors={"delta": torch.zeros(8)},
        tool_version="0.1.0",
        contracts={
            "contracts/preservation.yaml": _record_bytes(source_contract.to_dict()),
            "contracts/target.yaml": _record_bytes(source_contract.to_dict()),
        },
        provides=(source_contract.contract_id,),
        preserves=(source_contract.contract_id,),
    )
    source_patch_id = source_bundle.manifest.patch_id
    evidence = _evidence(
        source_patch_id=source_patch_id,
        source_base_hash=source_base_hash,
        target_base_hash=signature.signature_hash,
        target_contract_id=contract.contract_id,
        preservation_contract_id=contract.contract_id,
        old_contract_id=source_contract.contract_id,
    )
    bundle = create_patch_bundle(
        tmp_path / "rebased-patch",
        name="rebase-evidence-security",
        base_signature=signature.to_dict(),
        state_schema=state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("delta")}),
        tensors={"delta": torch.zeros(8)},
        tool_version="0.1.0",
        contracts={
            "contracts/preservation.yaml": _record_bytes(contract.to_dict()),
            "contracts/target.yaml": _record_bytes(contract.to_dict()),
        },
        supplemental_artifacts={
            "evidence/rebase.json": _record_bytes(evidence.to_dict()),
            "evidence/source-manifest.json": _record_bytes(source_bundle.manifest.to_dict()),
        },
        provides=(contract.contract_id,),
        preserves=(contract.contract_id,),
        rebased_from=source_patch_id,
        compiler_configuration={"mode": "semantic_recompile"},
    )
    return (
        bundle.path,
        source_patch_id,
        signature.signature_hash,
        contract.contract_id,
        source_contract.contract_id,
    )


@pytest.mark.parametrize(
    "alias",
    [
        r"evidence\rebase.json",
        "evidence/Rebase.json",
        "evidence/rebase.json.",
        "evidence/rebase.json ",
        "evidence/rebase.json:stream",
    ],
)
def test_bundle_manifest_rejects_noncanonical_rebase_artifact_path_alias(
    tmp_path: Path,
    alias: str,
) -> None:
    bundle, *_identities = _rebase_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = manifest["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes[alias] = artifact_hashes["evidence/rebase.json"]
    manifest_path.write_bytes(_record_bytes(manifest))

    with pytest.raises(ValueError):
        load_patch_bundle(bundle)


def test_rebased_bundle_requires_canonical_source_manifest(tmp_path: Path) -> None:
    bundle, *_identities = _rebase_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    source_manifest_path = bundle / "evidence" / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = manifest["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    source_value = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest_path.write_text(
        json.dumps(source_value, indent=2),
        encoding="utf-8",
    )
    artifact_hashes["evidence/source-manifest.json"] = sha256_file(source_manifest_path)
    manifest_path.write_bytes(_record_bytes(manifest))

    with pytest.raises(ValueError, match="canonical"):
        load_patch_bundle(bundle)


@pytest.mark.parametrize(
    "substitution",
    ["source-patch", "source-base", "source-contracts", "target-base", "contracts"],
)
def test_bundle_rejects_fully_rehashed_rebase_evidence_lineage_substitution(
    tmp_path: Path,
    substitution: str,
) -> None:
    bundle, _source_patch_id, _target_base_hash, _contract_id, source_contract_id = _rebase_bundle(
        tmp_path
    )
    evidence_path = bundle / "evidence" / "rebase.json"
    value = copy.deepcopy(json.loads(evidence_path.read_text(encoding="utf-8")))
    if substitution == "source-patch":
        value["source_patch_id"] = _digest(301)
    elif substitution == "source-base":
        value["source_base_hash"] = _digest(306)
    elif substitution == "source-contracts":
        value["old_patched_behavior"] = {_digest(307): 0.75}
    elif substitution == "target-base":
        value["target_base_hash"] = _digest(302)
    elif substitution == "contracts":
        # A valid source-side contract identity is still a substitution attack:
        # bundled evidence must bind to the packaged, target-base contract IDs.
        value["new_patched_behavior"] = {source_contract_id: 0.5}
        value["new_base_preservation"] = {f"{source_contract_id}:guards": 0.25}
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(substitution)
    _rehash(value)
    evidence_path.write_bytes(_record_bytes(value))

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["evidence/rebase.json"] = sha256_file(evidence_path)
    manifest_path.write_bytes(_record_bytes(manifest))

    with pytest.raises(ValueError):
        load_patch_bundle(bundle)
