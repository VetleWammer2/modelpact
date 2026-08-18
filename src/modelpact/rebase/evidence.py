"""Strict, content-addressed Rebase Evidence v1 records.

Rebase evidence is untrusted data.  The reader rejects malformed or
non-canonical JSON, verifies the record's content hash, enforces the closed v1
schema and resource limits, and checks claim/outcome consistency.  Callers that
know surrounding patch/base/contract identities can additionally bind them
through :class:`RebaseEvidenceExpectations`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast

from modelpact.contracts.parser import ContractLimits, loads_data, validate_data_shape
from modelpact.rebase.direct import RebaseCompatibility
from modelpact.status import RebaseClaim, VerificationOutcome
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, is_sha256_digest

MAX_REBASE_EVIDENCE_BYTES = 16 * 1024**2
MAX_REBASE_EVIDENCE_CONTRACTS = 100_000
MAX_REBASE_EVIDENCE_COMPLEXITY_METRICS = 1_024
MAX_REBASE_EVIDENCE_WARNINGS = 10_000
MAX_REBASE_EXECUTION_COUNT = 2**31 - 1
MAX_REBASE_COMPLEXITY_INTEGER = 2**63 - 1

_REBASE_EVIDENCE_LIMITS = ContractLimits(
    max_bytes=MAX_REBASE_EVIDENCE_BYTES,
    max_depth=16,
    max_nodes=500_000,
    max_string_length=4_096,
    max_object_keys=MAX_REBASE_EVIDENCE_CONTRACTS,
    max_objectives=1,
    max_assertions=1,
)
MAX_REBASE_REFERENCE_CHARS = 256
_PAYLOAD_FIELDS = frozenset(
    {
        "budget_exhausted",
        "claim",
        "compatibility",
        "direct_attempted",
        "direct_outcome",
        "new_base_preservation",
        "new_patched_behavior",
        "old_patched_behavior",
        "patch_complexity_after",
        "patch_complexity_before",
        "recompile_attempted",
        "recompile_restarts",
        "recompile_steps",
        "schema_version",
        "source_base_hash",
        "source_patch_id",
        "target_base_hash",
        "warnings",
    }
)
REBASE_EVIDENCE_FIELDS = _PAYLOAD_FIELDS | {"evidence_hash"}


class RebaseEvidenceError(ValueError):
    """Base class for invalid untrusted Rebase Evidence records."""


class RebaseEvidenceIntegrityError(RebaseEvidenceError):
    """Raised when hashes, lineage, or claim evidence are inconsistent."""


@dataclass(frozen=True, slots=True)
class RebaseEvidenceExpectations:
    evidence_hash: str | None = None
    source_patch_id: str | None = None
    source_base_hash: str | None = None
    target_base_hash: str | None = None
    claim: RebaseClaim | None = None
    source_contract_ids: frozenset[str] | None = None
    target_contract_ids: frozenset[str] | None = None
    preservation_contract_ids: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class RebaseEvidence:
    source_patch_id: str
    source_base_hash: str
    target_base_hash: str
    claim: RebaseClaim
    compatibility: str
    direct_attempted: bool
    direct_outcome: str | None
    recompile_attempted: bool
    recompile_steps: int
    recompile_restarts: int
    budget_exhausted: bool
    old_patched_behavior: Mapping[str, float] = field(default_factory=dict)
    new_patched_behavior: Mapping[str, float] = field(default_factory=dict)
    new_base_preservation: Mapping[str, float] = field(default_factory=dict)
    patch_complexity_before: Mapping[str, int | float] = field(default_factory=dict)
    patch_complexity_after: Mapping[str, int | float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # A frozen dataclass is not sufficient when callers retain mutable input
        # mappings.  Snapshot them so validation cannot race later mutation.
        for name in (
            "old_patched_behavior",
            "new_patched_behavior",
            "new_base_preservation",
            "patch_complexity_before",
            "patch_complexity_after",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_patch_id": self.source_patch_id,
            "source_base_hash": self.source_base_hash,
            "target_base_hash": self.target_base_hash,
            "claim": self.claim.value,
            "compatibility": self.compatibility,
            "direct_attempted": self.direct_attempted,
            "direct_outcome": self.direct_outcome,
            "recompile_attempted": self.recompile_attempted,
            "recompile_steps": self.recompile_steps,
            "recompile_restarts": self.recompile_restarts,
            "budget_exhausted": self.budget_exhausted,
            "old_patched_behavior": dict(sorted(self.old_patched_behavior.items())),
            "new_patched_behavior": dict(sorted(self.new_patched_behavior.items())),
            "new_base_preservation": dict(sorted(self.new_base_preservation.items())),
            "patch_complexity_before": dict(sorted(self.patch_complexity_before.items())),
            "patch_complexity_after": dict(sorted(self.patch_complexity_after.items())),
            "warnings": list(self.warnings),
        }

    @property
    def evidence_hash(self) -> str:
        return hash_canonical(self.payload())

    def to_dict(self) -> dict[str, object]:
        _validate_semantics(self)
        return {**self.payload(), "evidence_hash": self.evidence_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RebaseEvidence:
        return rebase_evidence_from_dict(value)


def _required_digest(value: object, name: str) -> str:
    if not is_sha256_digest(value):
        raise RebaseEvidenceError(f"{name} must be a lowercase sha256: digest")
    return cast(str, value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise RebaseEvidenceError(f"{name} must be a boolean")
    return value


def _execution_count(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_REBASE_EXECUTION_COUNT
    ):
        raise RebaseEvidenceError(
            f"{name} must be an integer from 0 through {MAX_REBASE_EXECUTION_COUNT}"
        )
    return value


def _reference(value: object, name: str) -> str:
    """Bound a contract reference without imposing a charset the schema lacks.

    A Behavior Contract identifier is an unconstrained bounded string, so this
    rejects only what makes a map key dangerous or ambiguous: path separators
    and traversal spellings, control characters, and surrounding whitespace.
    """

    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_REBASE_REFERENCE_CHARS
        or value in {".", ".."}
        or value != value.strip()
        or any(character in value for character in "/\\")
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RebaseEvidenceError(
            f"{name} must be a bounded reference without separators or control characters"
        )
    return value


def _margin_mapping(value: object, name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise RebaseEvidenceError(f"{name} must be an object")
    if len(value) > MAX_REBASE_EVIDENCE_CONTRACTS:
        raise RebaseEvidenceError(f"{name} exceeds the contract-reference limit")
    result: dict[str, float] = {}
    for raw_key, raw_margin in value.items():
        # Margin keys are contract references, not content addresses. The CLI
        # keys them by contract hash and role suffix, but the library API lets a
        # caller key them by the contract's declared identifier, so the schema
        # bounds their shape and leaves identity binding to
        # RebaseEvidenceExpectations, which knows the surrounding contract set.
        key = _reference(raw_key, f"{name} key")
        if not isinstance(raw_margin, float) or not math.isfinite(raw_margin):
            raise RebaseEvidenceError(f"{name}.{key} must be a finite JSON float")
        result[key] = raw_margin
    return MappingProxyType(result)


def _complexity_mapping(value: object, name: str) -> Mapping[str, int | float]:
    if not isinstance(value, Mapping):
        raise RebaseEvidenceError(f"{name} must be an object")
    if len(value) > MAX_REBASE_EVIDENCE_COMPLEXITY_METRICS:
        raise RebaseEvidenceError(f"{name} exceeds the complexity-metric limit")
    result: dict[str, int | float] = {}
    for raw_key, raw_metric in value.items():
        key = _reference(raw_key, f"{name} key")
        if isinstance(raw_metric, bool) or not isinstance(raw_metric, int | float):
            raise RebaseEvidenceError(f"{name}.{key} must be numeric")
        if isinstance(raw_metric, int) and raw_metric > MAX_REBASE_COMPLEXITY_INTEGER:
            raise RebaseEvidenceError(f"{name}.{key} exceeds the integer limit")
        if not math.isfinite(float(raw_metric)) or raw_metric < 0:
            raise RebaseEvidenceError(f"{name}.{key} must be finite and non-negative")
        result[key] = raw_metric
    return MappingProxyType(result)


def _warning_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RebaseEvidenceError("warnings must be an array of strings")
    if len(value) > MAX_REBASE_EVIDENCE_WARNINGS:
        raise RebaseEvidenceError("warnings exceeds the collection limit")
    warnings = tuple(cast(list[str], value))
    if len(warnings) != len(set(warnings)):
        raise RebaseEvidenceError("warnings cannot contain duplicates")
    return warnings


def _all_nonnegative(values: Mapping[str, float]) -> bool:
    return bool(values) and all(value >= 0.0 for value in values.values())


def _validate_semantics(evidence: RebaseEvidence) -> None:
    _required_digest(evidence.source_patch_id, "source_patch_id")
    _required_digest(evidence.source_base_hash, "source_base_hash")
    _required_digest(evidence.target_base_hash, "target_base_hash")
    if not isinstance(evidence.claim, RebaseClaim):
        raise RebaseEvidenceError("claim must be a RebaseClaim")
    _boolean(evidence.direct_attempted, "direct_attempted")
    _boolean(evidence.recompile_attempted, "recompile_attempted")
    _boolean(evidence.budget_exhausted, "budget_exhausted")
    try:
        compatibility = RebaseCompatibility(evidence.compatibility)
    except ValueError as error:
        raise RebaseEvidenceError(
            f"unsupported rebase compatibility: {evidence.compatibility!r}"
        ) from error
    try:
        direct_outcome = (
            None
            if evidence.direct_outcome is None
            else VerificationOutcome(evidence.direct_outcome)
        )
    except ValueError as error:
        raise RebaseEvidenceError(
            f"unsupported direct verification outcome: {evidence.direct_outcome!r}"
        ) from error

    _execution_count(evidence.recompile_steps, "recompile_steps")
    _execution_count(evidence.recompile_restarts, "recompile_restarts")
    _margin_mapping(evidence.old_patched_behavior, "old_patched_behavior")
    _margin_mapping(evidence.new_patched_behavior, "new_patched_behavior")
    _margin_mapping(evidence.new_base_preservation, "new_base_preservation")
    _complexity_mapping(evidence.patch_complexity_before, "patch_complexity_before")
    _complexity_mapping(evidence.patch_complexity_after, "patch_complexity_after")
    _warning_tuple(list(evidence.warnings))

    if evidence.direct_attempted:
        if compatibility is not RebaseCompatibility.DIRECT_PHYSICAL_TRANSFER:
            raise RebaseEvidenceIntegrityError(
                "a direct attempt requires DIRECT_PHYSICAL_TRANSFER compatibility"
            )
        if direct_outcome is None:
            raise RebaseEvidenceIntegrityError("a direct attempt must record its outcome")
    elif direct_outcome is not None:
        raise RebaseEvidenceIntegrityError("direct_outcome requires a direct attempt")

    if not evidence.recompile_attempted and (
        evidence.recompile_steps != 0
        or evidence.recompile_restarts != 0
        or evidence.budget_exhausted
    ):
        raise RebaseEvidenceIntegrityError(
            "unattempted recompilation must have zero counts and no exhausted budget"
        )
    if evidence.budget_exhausted and not evidence.recompile_attempted:
        raise RebaseEvidenceIntegrityError("budget exhaustion requires recompilation")
    if (
        evidence.recompile_attempted
        and compatibility is RebaseCompatibility.DIRECT_PHYSICAL_TRANSFER
        and not evidence.direct_attempted
    ):
        raise RebaseEvidenceIntegrityError(
            "recompilation after direct compatibility must retain the failed direct attempt"
        )

    incompatible = compatibility in {
        RebaseCompatibility.INCOMPATIBLE_TOKENIZER,
        RebaseCompatibility.INCOMPATIBLE_OUTPUT_SEMANTICS,
    }
    if evidence.claim is RebaseClaim.DIRECT_TRANSPLANT_VERIFIED:
        if (
            not evidence.direct_attempted
            or direct_outcome is not VerificationOutcome.PASS
            or evidence.recompile_attempted
            or (
                bool(evidence.old_patched_behavior)
                and not _all_nonnegative(evidence.old_patched_behavior)
            )
            or not _all_nonnegative(evidence.new_patched_behavior)
            or not _all_nonnegative(evidence.new_base_preservation)
            or dict(evidence.patch_complexity_before) != dict(evidence.patch_complexity_after)
        ):
            raise RebaseEvidenceIntegrityError(
                "DIRECT_TRANSPLANT_VERIFIED is inconsistent with its execution evidence"
            )
    elif evidence.claim is RebaseClaim.SEMANTIC_REBASE_VERIFIED:
        # recompile_restarts and patch_complexity_after are deliberately not
        # constrained here. BehavioralRecompileResult permits zero restarts and
        # defaults complexity to {}, so a third-party recompiler that converges
        # on its first attempt would otherwise produce a verified rebase that
        # cannot be serialized.
        if (
            incompatible
            or not evidence.recompile_attempted
            or evidence.recompile_steps <= 0
            or evidence.budget_exhausted
            or direct_outcome is VerificationOutcome.PASS
            or not _all_nonnegative(evidence.old_patched_behavior)
            or not _all_nonnegative(evidence.new_patched_behavior)
            or not _all_nonnegative(evidence.new_base_preservation)
        ):
            raise RebaseEvidenceIntegrityError(
                "SEMANTIC_REBASE_VERIFIED is inconsistent with its execution evidence"
            )
    elif evidence.claim is RebaseClaim.REBASE_FAILED:
        if not evidence.recompile_attempted or direct_outcome is VerificationOutcome.PASS:
            raise RebaseEvidenceIntegrityError(
                "REBASE_FAILED requires an attempted failed recompilation"
            )
    elif evidence.claim is RebaseClaim.REBASE_INCONCLUSIVE and (
        evidence.recompile_attempted or direct_outcome is VerificationOutcome.PASS
    ):
        raise RebaseEvidenceIntegrityError(
            "REBASE_INCONCLUSIVE cannot contain successful or attempted recompile evidence"
        )


def rebase_evidence_from_dict(value: Mapping[str, object]) -> RebaseEvidence:
    """Strictly parse and authenticate a materialized Rebase Evidence v1 object."""

    validate_data_shape(value, limits=_REBASE_EVIDENCE_LIMITS)
    unknown = set(value) - REBASE_EVIDENCE_FIELDS
    missing = REBASE_EVIDENCE_FIELDS - set(value)
    if unknown:
        raise RebaseEvidenceError(f"unknown Rebase Evidence v1 fields: {sorted(unknown)}")
    if missing:
        raise RebaseEvidenceError(f"missing Rebase Evidence v1 fields: {sorted(missing)}")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise RebaseEvidenceError("unsupported Rebase Evidence schema version")

    declared_hash = _required_digest(value.get("evidence_hash"), "evidence_hash")
    raw_payload = {name: value[name] for name in _PAYLOAD_FIELDS}
    observed_hash = hash_canonical(raw_payload)
    if declared_hash != observed_hash:
        raise RebaseEvidenceIntegrityError(
            "Rebase Evidence content hash mismatch: "
            f"declared {declared_hash}, observed {observed_hash}"
        )

    raw_claim = value.get("claim")
    if not isinstance(raw_claim, str):
        raise RebaseEvidenceError("claim must be a string")
    try:
        claim = RebaseClaim(raw_claim)
    except ValueError as error:
        raise RebaseEvidenceError(f"unsupported rebase claim: {raw_claim!r}") from error
    raw_compatibility = value.get("compatibility")
    if not isinstance(raw_compatibility, str):
        raise RebaseEvidenceError("compatibility must be a string")
    try:
        compatibility = RebaseCompatibility(raw_compatibility)
    except ValueError as error:
        raise RebaseEvidenceError(
            f"unsupported rebase compatibility: {raw_compatibility!r}"
        ) from error
    raw_direct_outcome = value.get("direct_outcome")
    if raw_direct_outcome is not None and not isinstance(raw_direct_outcome, str):
        raise RebaseEvidenceError("direct_outcome must be null or a string")
    if raw_direct_outcome is not None:
        try:
            VerificationOutcome(raw_direct_outcome)
        except ValueError as error:
            raise RebaseEvidenceError(
                f"unsupported direct verification outcome: {raw_direct_outcome!r}"
            ) from error

    evidence = RebaseEvidence(
        source_patch_id=_required_digest(value.get("source_patch_id"), "source_patch_id"),
        source_base_hash=_required_digest(value.get("source_base_hash"), "source_base_hash"),
        target_base_hash=_required_digest(value.get("target_base_hash"), "target_base_hash"),
        claim=claim,
        compatibility=compatibility.value,
        direct_attempted=_boolean(value.get("direct_attempted"), "direct_attempted"),
        direct_outcome=raw_direct_outcome,
        recompile_attempted=_boolean(value.get("recompile_attempted"), "recompile_attempted"),
        recompile_steps=_execution_count(value.get("recompile_steps"), "recompile_steps"),
        recompile_restarts=_execution_count(value.get("recompile_restarts"), "recompile_restarts"),
        budget_exhausted=_boolean(value.get("budget_exhausted"), "budget_exhausted"),
        old_patched_behavior=_margin_mapping(
            value.get("old_patched_behavior"), "old_patched_behavior"
        ),
        new_patched_behavior=_margin_mapping(
            value.get("new_patched_behavior"), "new_patched_behavior"
        ),
        new_base_preservation=_margin_mapping(
            value.get("new_base_preservation"), "new_base_preservation"
        ),
        patch_complexity_before=_complexity_mapping(
            value.get("patch_complexity_before"), "patch_complexity_before"
        ),
        patch_complexity_after=_complexity_mapping(
            value.get("patch_complexity_after"), "patch_complexity_after"
        ),
        warnings=_warning_tuple(value.get("warnings")),
    )
    _validate_semantics(evidence)
    if evidence.evidence_hash != declared_hash:
        raise RebaseEvidenceIntegrityError(
            "Rebase Evidence schema normalization would change its content hash"
        )
    return evidence


def loads_rebase_evidence(text: str | bytes) -> RebaseEvidence:
    """Parse canonical, resource-bounded JSON into Rebase Evidence v1."""

    value = loads_data(
        text,
        format="json",
        limits=_REBASE_EVIDENCE_LIMITS,
        require_canonical=True,
    )
    if not isinstance(value, Mapping):
        raise RebaseEvidenceError("Rebase Evidence root must be an object")
    return rebase_evidence_from_dict(cast(Mapping[str, object], value))


def validate_rebase_evidence(
    evidence: RebaseEvidence,
    *,
    expectations: RebaseEvidenceExpectations | None = None,
) -> None:
    """Revalidate a record and bind it to identities supplied by its context."""

    reparsed = rebase_evidence_from_dict(evidence.to_dict())
    expected = expectations or RebaseEvidenceExpectations()
    scalar_checks: tuple[tuple[str, object | None, object], ...] = (
        ("evidence_hash", expected.evidence_hash, reparsed.evidence_hash),
        ("source_patch_id", expected.source_patch_id, reparsed.source_patch_id),
        ("source_base_hash", expected.source_base_hash, reparsed.source_base_hash),
        ("target_base_hash", expected.target_base_hash, reparsed.target_base_hash),
        ("claim", expected.claim, reparsed.claim),
    )
    for name, required, observed in scalar_checks:
        if required is not None and required != observed:
            raise RebaseEvidenceIntegrityError(
                f"{name} mismatch: expected {required}, observed {observed}"
            )
    observed_source_contracts = set(reparsed.old_patched_behavior)
    if expected.source_contract_ids is not None:
        required_source_contracts = set(expected.source_contract_ids)
        source_matches = (
            observed_source_contracts == required_source_contracts
            if reparsed.claim is RebaseClaim.SEMANTIC_REBASE_VERIFIED
            else observed_source_contracts.issubset(required_source_contracts)
        )
        if not source_matches:
            raise RebaseEvidenceIntegrityError(
                "source contracts mismatch: expected "
                f"{sorted(required_source_contracts)}, observed "
                f"{sorted(observed_source_contracts)}"
            )
    collection_checks = (
        ("target contracts", expected.target_contract_ids, set(reparsed.new_patched_behavior)),
        (
            "preservation contracts",
            expected.preservation_contract_ids,
            set(reparsed.new_base_preservation),
        ),
    )
    for name, required, observed in collection_checks:
        if required is not None and set(required) != observed:
            raise RebaseEvidenceIntegrityError(
                f"{name} mismatch: expected {sorted(required)}, observed {sorted(observed)}"
            )


def read_rebase_evidence(
    path: str | Path,
    *,
    expectations: RebaseEvidenceExpectations | None = None,
) -> RebaseEvidence:
    """Read a regular non-symlink Rebase Evidence file and validate its context."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RebaseEvidenceError("Rebase Evidence must be a regular file")
    try:
        size = source.stat().st_size
    except OSError as error:
        raise RebaseEvidenceError(f"cannot stat Rebase Evidence: {error}") from error
    if size > MAX_REBASE_EVIDENCE_BYTES:
        raise RebaseEvidenceError("Rebase Evidence exceeds the size limit")
    try:
        evidence = loads_rebase_evidence(source.read_bytes())
    except OSError as error:
        raise RebaseEvidenceError(f"cannot read Rebase Evidence: {error}") from error
    validate_rebase_evidence(evidence, expectations=expectations)
    return evidence


def write_rebase_evidence(
    evidence: RebaseEvidence,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    """Validate and atomically write canonical Rebase Evidence v1."""

    validate_rebase_evidence(evidence)
    encoded = (
        canonical_dumps(evidence.to_dict(), max_depth=_REBASE_EVIDENCE_LIMITS.max_depth) + "\n"
    )
    if len(encoded.encode("utf-8")) > MAX_REBASE_EVIDENCE_BYTES:
        raise RebaseEvidenceError("Rebase Evidence exceeds the size limit")
    atomic_write_text(
        path,
        encoded,
        overwrite=overwrite,
    )


__all__ = [
    "MAX_REBASE_EVIDENCE_BYTES",
    "MAX_REBASE_EVIDENCE_CONTRACTS",
    "MAX_REBASE_REFERENCE_CHARS",
    "REBASE_EVIDENCE_FIELDS",
    "RebaseEvidence",
    "RebaseEvidenceError",
    "RebaseEvidenceExpectations",
    "RebaseEvidenceIntegrityError",
    "loads_rebase_evidence",
    "read_rebase_evidence",
    "rebase_evidence_from_dict",
    "validate_rebase_evidence",
    "write_rebase_evidence",
]
