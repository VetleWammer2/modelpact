"""Content-addressed Verification Certificate v1.

Certificates are untrusted evidence records.  Validation checks schema,
content hashes, claim/evidence consistency, caller-provided identities, and
optionally every referenced artifact.  It never executes bundle content.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch

from modelpact import __version__
from modelpact.contracts.ast import AssertionType, BehaviorContract
from modelpact.contracts.parser import ContractLimits, loads_data
from modelpact.status import (
    AuditClaim,
    CompositionClaim,
    PatchClaim,
    RebaseClaim,
    VerificationOutcome,
)
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_file
from modelpact.util.paths import resolve_inside, safe_relative_path
from modelpact.verify.engine import VerificationReport

_CERTIFICATE_LIMITS = ContractLimits(
    max_bytes=16 * 1024 * 1024,
    max_depth=64,
    max_nodes=1_000_000,
    max_string_length=2_000_000,
    max_object_keys=100_000,
    max_objectives=100_000,
    max_assertions=100_000,
)
_HASH_LENGTH = 71
_ALLOWED_CLAIMS = frozenset(
    item.value
    for enum_type in (PatchClaim, CompositionClaim, RebaseClaim, AuditClaim)
    for item in enum_type
)


class CertificateError(ValueError):
    pass


class CertificateIntegrityError(CertificateError):
    pass


@dataclass(frozen=True, slots=True)
class CertificateExpectations:
    certificate_hash: str | None = None
    patch_id: str | None = None
    base_signature: str | None = None
    tokenizer_hash: str | None = None
    contract_hashes: Mapping[str, str] = field(default_factory=dict)
    probe_hashes: Mapping[str, str] = field(default_factory=dict)
    checkpoint_hashes: Mapping[str, str] = field(default_factory=dict)
    verification_result_hash: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationCertificate:
    schema_version: int
    modelpact_version: str
    patch_id: str
    base_signature: str
    model_adapter_id: str
    checkpoint_hashes: Mapping[str, str]
    tokenizer_hash: str
    contract_hashes: Mapping[str, str]
    probe_hashes: Mapping[str, str]
    verification_policy_hash: str
    generation_policy: Mapping[str, object]
    random_seeds: Mapping[str, object]
    compile_objectives: tuple[Mapping[str, object], ...]
    target_assertions: tuple[Mapping[str, object], ...]
    guard_assertions: tuple[Mapping[str, object], ...]
    sealed_holdout_result: Mapping[str, object]
    free_generation_results: tuple[Mapping[str, object], ...]
    prompt_level_metrics: tuple[Mapping[str, object], ...]
    statistical_intervals: tuple[Mapping[str, object], ...]
    counterexample_search: Mapping[str, object]
    patch_structure: Mapping[str, object]
    minimization_result: Mapping[str, object]
    composition_result: Mapping[str, object]
    interaction_diagnostics: Mapping[str, object]
    rebase_result: Mapping[str, object]
    environment_identity: Mapping[str, object]
    artifact_hashes: Mapping[str, str]
    verification_outcome: VerificationOutcome
    verification_result_hash: str
    claims: tuple[str, ...]
    warnings: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    compatibility_errors: tuple[str, ...]
    certificate_hash: str

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "modelpact_version": self.modelpact_version,
            "patch_id": self.patch_id,
            "base_signature": self.base_signature,
            "model_adapter_id": self.model_adapter_id,
            "checkpoint_hashes": dict(sorted(self.checkpoint_hashes.items())),
            "tokenizer_hash": self.tokenizer_hash,
            "contract_hashes": dict(sorted(self.contract_hashes.items())),
            "probe_hashes": dict(sorted(self.probe_hashes.items())),
            "verification_policy_hash": self.verification_policy_hash,
            "generation_policy": dict(self.generation_policy),
            "random_seeds": dict(self.random_seeds),
            "compile_objectives": [dict(item) for item in self.compile_objectives],
            "target_assertions": [dict(item) for item in self.target_assertions],
            "guard_assertions": [dict(item) for item in self.guard_assertions],
            "sealed_holdout_result": dict(self.sealed_holdout_result),
            "free_generation_results": [dict(item) for item in self.free_generation_results],
            "prompt_level_metrics": [dict(item) for item in self.prompt_level_metrics],
            "statistical_intervals": [dict(item) for item in self.statistical_intervals],
            "counterexample_search": dict(self.counterexample_search),
            "patch_structure": dict(self.patch_structure),
            "minimization_result": dict(self.minimization_result),
            "composition_result": dict(self.composition_result),
            "interaction_diagnostics": dict(self.interaction_diagnostics),
            "rebase_result": dict(self.rebase_result),
            "environment_identity": dict(self.environment_identity),
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "verification_outcome": self.verification_outcome.value,
            "verification_result_hash": self.verification_result_hash,
            "claims": list(self.claims),
            "warnings": list(self.warnings),
            "unsupported_claims": list(self.unsupported_claims),
            "compatibility_errors": list(self.compatibility_errors),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "certificate_hash": self.certificate_hash}

    def canonical_json(self) -> str:
        return canonical_dumps(self.to_dict(), max_depth=_CERTIFICATE_LIMITS.max_depth)


def _environment_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "byteorder": sys.byteorder,
    }


def _prompt_metrics(report: VerificationReport) -> tuple[Mapping[str, object], ...]:
    groups = (
        ("target", report.target_results),
        ("guard", report.guard_results),
        ("holdout_target", report.holdout_target_results),
        ("holdout_guard", report.holdout_guard_results),
    )
    result = []
    for role, evaluations in groups:
        for evaluation in evaluations:
            for item in evaluation.prompt_metrics:
                result.append(
                    {
                        "role": role,
                        "assertion_id": evaluation.assertion_id,
                        **item.to_dict(),
                    }
                )
    return tuple(result)


def _intervals(report: VerificationReport) -> tuple[Mapping[str, object], ...]:
    result = []
    for role, evaluations in (
        ("target", report.target_results),
        ("guard", report.guard_results),
        ("holdout_target", report.holdout_target_results),
        ("holdout_guard", report.holdout_guard_results),
    ):
        for evaluation in evaluations:
            if evaluation.confidence_interval is not None:
                result.append(
                    {
                        "role": role,
                        "assertion_id": evaluation.assertion_id,
                        **evaluation.confidence_interval.to_dict(),
                    }
                )
    return tuple(result)


def _derived_claims(
    report: VerificationReport,
    *,
    objectives_optimized: bool,
    minimized_within_budget: bool,
) -> tuple[str, ...]:
    claims: list[str] = []
    if not report.compatibility_errors:
        claims.append(PatchClaim.BASE_COMPATIBLE.value)
    if objectives_optimized:
        claims.append(PatchClaim.TARGET_OBJECTIVE_OPTIMIZED.value)
    if report.target_results and all(
        item.outcome is VerificationOutcome.PASS for item in report.target_results
    ):
        claims.append(PatchClaim.TARGET_ASSERTIONS_VERIFIED.value)
    if report.guard_results and all(
        item.outcome is VerificationOutcome.PASS for item in report.guard_results
    ):
        claims.append(PatchClaim.PRESERVATION_ASSERTIONS_VERIFIED.value)
    if report.holdout_outcome is VerificationOutcome.PASS and (
        report.holdout_target_results or report.holdout_guard_results
    ):
        claims.append(PatchClaim.SEALED_HOLDOUT_VERIFIED.value)
    generation_results = [
        *report.target_results,
        *report.guard_results,
        *report.holdout_target_results,
        *report.holdout_guard_results,
    ]
    generative_types = {
        AssertionType.EXACT_MATCH,
        AssertionType.NORMALIZED_EXACT_MATCH,
        AssertionType.REGULAR_EXPRESSION,
        AssertionType.JSON_PARSE,
        AssertionType.JSON_SCHEMA,
        AssertionType.FREE_GENERATION_MATCH,
        AssertionType.GENERATION_LENGTH,
    }
    relevant = [item for item in generation_results if item.assertion_type in generative_types]
    if (
        report.free_generation_records
        and relevant
        and all(item.outcome is VerificationOutcome.PASS for item in relevant)
    ):
        claims.append(PatchClaim.FREE_GENERATION_VERIFIED.value)
    if minimized_within_budget:
        claims.append(PatchClaim.PATCH_MINIMIZED_WITHIN_BUDGET.value)
    return tuple(sorted(claims))


def build_certificate(
    report: VerificationReport,
    contract: BehaviorContract,
    *,
    patch_id: str,
    checkpoint_hashes: Mapping[str, str],
    artifact_hashes: Mapping[str, str],
    verification_policy: Mapping[str, object] | None = None,
    counterexample_search: Mapping[str, object] | None = None,
    patch_structure: Mapping[str, object] | None = None,
    minimization_result: Mapping[str, object] | None = None,
    composition_result: Mapping[str, object] | None = None,
    interaction_diagnostics: Mapping[str, object] | None = None,
    rebase_result: Mapping[str, object] | None = None,
    environment_identity: Mapping[str, object] | None = None,
    objectives_optimized: bool = False,
    minimized_within_budget: bool = False,
    additional_warnings: Sequence[str] = (),
) -> VerificationCertificate:
    if report.contract_hash != contract.contract_id:
        raise CertificateIntegrityError("verification report and contract hashes differ")
    if report.identity.tokenizer_hash != (
        contract.model_requirements.tokenizer_hash or report.identity.tokenizer_hash
    ):
        raise CertificateIntegrityError("verification identity does not meet contract tokenizer")
    policy = verification_policy or {
        "statistics": contract.statistics.to_dict(),
        "generation": contract.generation.to_dict(),
    }
    holdout = {
        "outcome": report.holdout_outcome.value,
        "targets": [item.to_dict() for item in report.holdout_target_results],
        "guards": [item.to_dict() for item in report.holdout_guard_results],
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "modelpact_version": __version__,
        "patch_id": patch_id,
        "base_signature": report.identity.base_signature,
        "model_adapter_id": report.identity.adapter_id,
        "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
        "tokenizer_hash": report.identity.tokenizer_hash,
        "contract_hashes": {contract.id: contract.contract_id},
        "probe_hashes": dict(sorted(report.probe_hashes.items())),
        "verification_policy_hash": hash_canonical(policy),
        "generation_policy": contract.generation.to_dict(),
        "random_seeds": {
            "bootstrap_seed": contract.statistics.bootstrap_seed,
            "generation_seeds": list(contract.generation.seeds),
        },
        "compile_objectives": [item.to_dict() for item in contract.objectives],
        "target_assertions": [item.to_dict() for item in report.target_results],
        "guard_assertions": [item.to_dict() for item in report.guard_results],
        "sealed_holdout_result": holdout,
        "free_generation_results": [item.to_dict() for item in report.free_generation_records],
        "prompt_level_metrics": list(_prompt_metrics(report)),
        "statistical_intervals": list(_intervals(report)),
        "counterexample_search": dict(counterexample_search or {"outcome": "NOT_EXECUTED"}),
        "patch_structure": dict(patch_structure or {}),
        "minimization_result": dict(minimization_result or {"outcome": "UNMINIMIZED"}),
        "composition_result": dict(composition_result or {"outcome": "NOT_APPLICABLE"}),
        "interaction_diagnostics": dict(interaction_diagnostics or {}),
        "rebase_result": dict(rebase_result or {"outcome": "NOT_APPLICABLE"}),
        "environment_identity": dict(environment_identity or _environment_identity()),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "verification_outcome": report.outcome.value,
        "verification_result_hash": report.result_hash,
        "claims": list(
            _derived_claims(
                report,
                objectives_optimized=objectives_optimized,
                minimized_within_budget=minimized_within_budget,
            )
        ),
        "warnings": sorted({*report.warnings, *additional_warnings}),
        "unsupported_claims": sorted(set(report.unsupported_claims)),
        "compatibility_errors": list(report.compatibility_errors),
    }
    certificate_hash = hash_canonical(payload)
    return certificate_from_dict({**payload, "certificate_hash": certificate_hash})


_FIELDS = {
    "schema_version",
    "modelpact_version",
    "patch_id",
    "base_signature",
    "model_adapter_id",
    "checkpoint_hashes",
    "tokenizer_hash",
    "contract_hashes",
    "probe_hashes",
    "verification_policy_hash",
    "generation_policy",
    "random_seeds",
    "compile_objectives",
    "target_assertions",
    "guard_assertions",
    "sealed_holdout_result",
    "free_generation_results",
    "prompt_level_metrics",
    "statistical_intervals",
    "counterexample_search",
    "patch_structure",
    "minimization_result",
    "composition_result",
    "interaction_diagnostics",
    "rebase_result",
    "environment_identity",
    "artifact_hashes",
    "verification_outcome",
    "verification_result_hash",
    "claims",
    "warnings",
    "unsupported_claims",
    "compatibility_errors",
    "certificate_hash",
}


def _required_string(data: Mapping[str, object], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value or len(value) > 1_000_000 or "\x00" in value:
        raise CertificateError(f"{name} must be a non-empty bounded string")
    return value


def _mapping(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise CertificateError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CertificateError(f"{name} keys must be strings")
    return cast(Mapping[str, object], value)


def _string_mapping(data: Mapping[str, object], name: str) -> dict[str, str]:
    value = _mapping(data, name)
    if any(not isinstance(item, str) for item in value.values()):
        raise CertificateError(f"{name} values must be strings")
    return {key: cast(str, item) for key, item in value.items()}


def _object_tuple(data: Mapping[str, object], name: str) -> tuple[Mapping[str, object], ...]:
    value = data.get(name)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise CertificateError(f"{name} must be an array of objects")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _string_tuple(data: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = data.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CertificateError(f"{name} must be an array of strings")
    if len(value) != len(set(value)):
        raise CertificateError(f"{name} cannot contain duplicates")
    return tuple(cast(list[str], value))


def _is_sha256(value: str) -> bool:
    return (
        len(value) == _HASH_LENGTH
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _validate_hash(value: str, name: str) -> None:
    if not _is_sha256(value):
        raise CertificateError(f"{name} must be a lowercase sha256: digest")


def certificate_from_dict(value: Mapping[str, object]) -> VerificationCertificate:
    """Parse a fully materialized mapping and verify its self-hash."""

    unknown = set(value) - _FIELDS
    missing = _FIELDS - set(value)
    if unknown:
        raise CertificateError("unknown certificate field(s): " + ", ".join(sorted(unknown)))
    if missing:
        raise CertificateError("missing certificate field(s): " + ", ".join(sorted(missing)))
    if value.get("schema_version") != 1:
        raise CertificateError("only Verification Certificate schema_version 1 is supported")
    outcome_text = _required_string(value, "verification_outcome")
    try:
        outcome = VerificationOutcome(outcome_text)
    except ValueError as error:
        raise CertificateError(f"unknown verification outcome {outcome_text!r}") from error
    checkpoint_hashes = _string_mapping(value, "checkpoint_hashes")
    contract_hashes = _string_mapping(value, "contract_hashes")
    probe_hashes = _string_mapping(value, "probe_hashes")
    artifact_hashes = _string_mapping(value, "artifact_hashes")
    hashes = {
        "tokenizer_hash": _required_string(value, "tokenizer_hash"),
        "verification_policy_hash": _required_string(value, "verification_policy_hash"),
        "verification_result_hash": _required_string(value, "verification_result_hash"),
        "certificate_hash": _required_string(value, "certificate_hash"),
        **{f"checkpoint_hashes.{key}": item for key, item in checkpoint_hashes.items()},
        **{f"contract_hashes.{key}": item for key, item in contract_hashes.items()},
        **{f"probe_hashes.{key}": item for key, item in probe_hashes.items()},
        **{f"artifact_hashes.{key}": item for key, item in artifact_hashes.items()},
    }
    for name, digest in hashes.items():
        _validate_hash(digest, name)
    for path in artifact_hashes:
        safe_relative_path(path)
    claims = _string_tuple(value, "claims")
    unsupported_claims = _string_tuple(value, "unsupported_claims")
    unknown_claims = (set(claims) | set(unsupported_claims)) - _ALLOWED_CLAIMS
    if unknown_claims:
        raise CertificateError("unknown claim(s): " + ", ".join(sorted(unknown_claims)))
    overlap = set(claims) & set(unsupported_claims)
    if overlap:
        raise CertificateError(
            "claims cannot simultaneously be supported and unsupported: "
            + ", ".join(sorted(overlap))
        )
    certificate_hash = hashes["certificate_hash"]
    payload = dict(value)
    del payload["certificate_hash"]
    observed_hash = hash_canonical(payload)
    if observed_hash != certificate_hash:
        raise CertificateIntegrityError(
            "certificate content hash mismatch: "
            f"declared {certificate_hash}, observed {observed_hash}"
        )
    certificate = VerificationCertificate(
        schema_version=1,
        modelpact_version=_required_string(value, "modelpact_version"),
        patch_id=_required_string(value, "patch_id"),
        base_signature=_required_string(value, "base_signature"),
        model_adapter_id=_required_string(value, "model_adapter_id"),
        checkpoint_hashes=checkpoint_hashes,
        tokenizer_hash=hashes["tokenizer_hash"],
        contract_hashes=contract_hashes,
        probe_hashes=probe_hashes,
        verification_policy_hash=hashes["verification_policy_hash"],
        generation_policy=_mapping(value, "generation_policy"),
        random_seeds=_mapping(value, "random_seeds"),
        compile_objectives=_object_tuple(value, "compile_objectives"),
        target_assertions=_object_tuple(value, "target_assertions"),
        guard_assertions=_object_tuple(value, "guard_assertions"),
        sealed_holdout_result=_mapping(value, "sealed_holdout_result"),
        free_generation_results=_object_tuple(value, "free_generation_results"),
        prompt_level_metrics=_object_tuple(value, "prompt_level_metrics"),
        statistical_intervals=_object_tuple(value, "statistical_intervals"),
        counterexample_search=_mapping(value, "counterexample_search"),
        patch_structure=_mapping(value, "patch_structure"),
        minimization_result=_mapping(value, "minimization_result"),
        composition_result=_mapping(value, "composition_result"),
        interaction_diagnostics=_mapping(value, "interaction_diagnostics"),
        rebase_result=_mapping(value, "rebase_result"),
        environment_identity=_mapping(value, "environment_identity"),
        artifact_hashes=artifact_hashes,
        verification_outcome=outcome,
        verification_result_hash=hashes["verification_result_hash"],
        claims=claims,
        warnings=_string_tuple(value, "warnings"),
        unsupported_claims=unsupported_claims,
        compatibility_errors=_string_tuple(value, "compatibility_errors"),
        certificate_hash=certificate_hash,
    )
    _validate_claim_evidence(certificate)
    return certificate


def _all_pass(assertions: Sequence[Mapping[str, object]]) -> bool:
    outcomes_pass = all(
        item.get("outcome") == VerificationOutcome.PASS.value for item in assertions
    )
    return bool(assertions) and outcomes_pass


def _validate_claim_evidence(certificate: VerificationCertificate) -> None:
    claims = set(certificate.claims)
    checks = (
        (
            PatchClaim.BASE_COMPATIBLE.value,
            not certificate.compatibility_errors,
            "compatibility errors are present",
        ),
        (
            PatchClaim.TARGET_ASSERTIONS_VERIFIED.value,
            _all_pass(certificate.target_assertions),
            "target assertion evidence is absent or not all passing",
        ),
        (
            PatchClaim.PRESERVATION_ASSERTIONS_VERIFIED.value,
            _all_pass(certificate.guard_assertions),
            "guard assertion evidence is absent or not all passing",
        ),
        (
            PatchClaim.SEALED_HOLDOUT_VERIFIED.value,
            certificate.sealed_holdout_result.get("outcome") == VerificationOutcome.PASS.value,
            "sealed holdout outcome is not PASS",
        ),
        (
            PatchClaim.FREE_GENERATION_VERIFIED.value,
            bool(certificate.free_generation_results),
            "free-generation evidence is absent",
        ),
    )
    for claim, supported, reason in checks:
        if claim in claims and not supported:
            raise CertificateIntegrityError(f"claim {claim} is unsupported: {reason}")
    if certificate.verification_outcome is VerificationOutcome.PASS and (
        not _all_pass(certificate.target_assertions) and not _all_pass(certificate.guard_assertions)
    ):
        raise CertificateIntegrityError("PASS certificate contains no passing assertion group")


def loads_certificate(text: str | bytes) -> VerificationCertificate:
    value = loads_data(text, format="json", limits=_CERTIFICATE_LIMITS)
    if not isinstance(value, Mapping):
        raise CertificateError("certificate root must be an object")
    return certificate_from_dict(cast(Mapping[str, object], value))


def read_certificate(path: str | Path) -> VerificationCertificate:
    source = Path(path)
    if source.stat().st_size > _CERTIFICATE_LIMITS.max_bytes:
        raise CertificateError("certificate exceeds size limit")
    return loads_certificate(source.read_bytes())


def write_certificate(
    certificate: VerificationCertificate,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    # Revalidate immediately before writing in case a caller retained and
    # mutated one of the mapping objects used to construct the dataclass.
    validated = certificate_from_dict(certificate.to_dict())
    atomic_write_text(
        path,
        validated.canonical_json() + "\n",
        encoding="utf-8",
        overwrite=overwrite,
    )


def validate_certificate(
    certificate: VerificationCertificate,
    *,
    expectations: CertificateExpectations | None = None,
    artifact_root: str | Path | None = None,
) -> None:
    """Recompute certificate/artifact hashes and compare external identities."""

    reparsed = certificate_from_dict(certificate.to_dict())
    expected = expectations or CertificateExpectations()
    scalar_checks = (
        ("certificate_hash", expected.certificate_hash, reparsed.certificate_hash),
        ("patch_id", expected.patch_id, reparsed.patch_id),
        ("base_signature", expected.base_signature, reparsed.base_signature),
        ("tokenizer_hash", expected.tokenizer_hash, reparsed.tokenizer_hash),
        (
            "verification_result_hash",
            expected.verification_result_hash,
            reparsed.verification_result_hash,
        ),
    )
    for name, required, observed in scalar_checks:
        if required is not None and required != observed:
            raise CertificateIntegrityError(
                f"{name} mismatch: expected {required}, observed {observed}"
            )
    mapping_checks: tuple[tuple[str, Mapping[str, str], Mapping[str, str]], ...] = (
        ("contract_hashes", expected.contract_hashes, reparsed.contract_hashes),
        ("probe_hashes", expected.probe_hashes, reparsed.probe_hashes),
        ("checkpoint_hashes", expected.checkpoint_hashes, reparsed.checkpoint_hashes),
    )
    for mapping_name, required_mapping, observed_mapping in mapping_checks:
        for key, digest in required_mapping.items():
            if observed_mapping.get(key) != digest:
                raise CertificateIntegrityError(
                    f"{mapping_name}.{key} mismatch: expected {digest}, "
                    f"observed {observed_mapping.get(key)}"
                )
    if artifact_root is not None:
        for relative, declared_hash in sorted(reparsed.artifact_hashes.items()):
            path = resolve_inside(artifact_root, relative)
            if not path.is_file():
                raise CertificateIntegrityError(f"referenced artifact is missing: {relative}")
            observed_hash = sha256_file(path)
            if observed_hash != declared_hash:
                raise CertificateIntegrityError(
                    f"artifact hash mismatch for {relative}: "
                    f"declared {declared_hash}, observed {observed_hash}"
                )


__all__ = [
    "CertificateError",
    "CertificateExpectations",
    "CertificateIntegrityError",
    "VerificationCertificate",
    "build_certificate",
    "certificate_from_dict",
    "loads_certificate",
    "read_certificate",
    "validate_certificate",
    "write_certificate",
]
