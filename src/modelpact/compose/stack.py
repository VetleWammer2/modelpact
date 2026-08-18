"""Deterministic lineage and declarative patch-stack resolution records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from modelpact.contracts.parser import ContractLimits, loads_data, validate_data_shape
from modelpact.util.hashing import is_sha256_digest

MAX_STACK_LOCK_BYTES = 16 * 1024**2
MAX_STACK_LOCK_PATCHES = 4_096
MAX_STACK_LOCK_CONTRACTS = 100_000
MAX_STACK_LOCK_NODES = 150_000
MAX_STACK_LOCK_STRING_LENGTH = 4_096
MAX_STACK_LOCK_OBJECT_KEYS = 10_000
STACK_LOCK_LIMITS = ContractLimits(
    max_bytes=MAX_STACK_LOCK_BYTES,
    max_depth=16,
    max_nodes=MAX_STACK_LOCK_NODES,
    max_string_length=MAX_STACK_LOCK_STRING_LENGTH,
    max_object_keys=MAX_STACK_LOCK_OBJECT_KEYS,
    max_objectives=1,
    max_assertions=1,
)
STACK_LOCK_FIELDS = frozenset(
    {
        "audit_hash",
        "base_hash",
        "certificate_hash",
        "contract_hashes",
        "patch_hashes",
        "resolution",
        "resolved_artifact_hash",
        "schema_version",
        "verification_policy_hash",
    }
)


@dataclass(frozen=True, slots=True)
class PatchLineage:
    parent_patches: tuple[str, ...] = ()
    merged_from: tuple[str, ...] = ()
    rebased_from: str | None = None
    source_diff: str | None = None

    def normalized(self) -> PatchLineage:
        return PatchLineage(
            parent_patches=tuple(sorted(set(self.parent_patches))),
            merged_from=tuple(sorted(set(self.merged_from))),
            rebased_from=self.rebased_from,
            source_diff=self.source_diff,
        )


@dataclass(frozen=True, slots=True)
class PatchReference:
    patch_id: str
    patch_hash: str
    base_hash: str
    contract_hashes: tuple[str, ...]
    artifact_hash: str
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    lineage: PatchLineage = field(default_factory=PatchLineage)

    def __post_init__(self) -> None:
        required = (
            self.patch_id,
            self.patch_hash,
            self.base_hash,
            self.artifact_hash,
        )
        if any(not value for value in required):
            raise ValueError("patch reference identities must not be empty")
        for name, values in (
            ("contract_hashes", self.contract_hashes),
            ("requires", self.requires),
            ("provides", self.provides),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"patch reference {name} must be sorted and unique")

    @property
    def provided_contracts(self) -> tuple[str, ...]:
        """Contracts capable of satisfying another patch's requirements."""

        return self.provides


class StackResolutionKind(StrEnum):
    NAIVE_ADDITIVE_STACK = "NAIVE_ADDITIVE_STACK"
    VERIFIED_COMPOSITE_PATCH = "VERIFIED_COMPOSITE_PATCH"
    PARTIALLY_RESOLVED_STACK = "PARTIALLY_RESOLVED_STACK"
    STATIC_CONTRADICTION = "STATIC_CONTRADICTION"
    EMPIRICAL_FAILURE = "EMPIRICAL_FAILURE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class StackResolutionExecution:
    kind: StackResolutionKind
    resolved_artifact_hash: str | None
    verification_policy_hash: str
    union_contract_hash: str
    certificate_hash: str | None = None
    audit_hash: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.verification_policy_hash or not self.union_contract_hash:
            raise ValueError("stack resolution must pin policy and union-contract hashes")
        if (
            self.kind
            in {
                StackResolutionKind.NAIVE_ADDITIVE_STACK,
                StackResolutionKind.VERIFIED_COMPOSITE_PATCH,
            }
            and not self.resolved_artifact_hash
        ):
            raise ValueError("successful stack resolution must pin a resolved artifact")


@dataclass(frozen=True, slots=True)
class StackResolutionRequest:
    base_hash: str
    patches: tuple[PatchReference, ...]
    repair_conflicts: bool
    subset_audit_budget: int


class StackResolver(Protocol):
    def __call__(self, request: StackResolutionRequest) -> StackResolutionExecution: ...


@dataclass(frozen=True, slots=True)
class StackLock:
    schema_version: int
    base_hash: str
    patch_hashes: Mapping[str, str]
    contract_hashes: tuple[str, ...]
    resolved_artifact_hash: str | None
    verification_policy_hash: str
    resolution: StackResolutionKind
    certificate_hash: str | None
    audit_hash: str | None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or type(self.schema_version) is not int:
            raise ValueError("unsupported Patch Stack Lockfile schema version")
        if self.schema_version != 1:
            raise ValueError("unsupported Patch Stack Lockfile schema version")
        if not isinstance(self.resolution, StackResolutionKind):
            raise ValueError("lockfile resolution must be a StackResolutionKind")
        for name, digest in (
            ("base_hash", self.base_hash),
            ("verification_policy_hash", self.verification_policy_hash),
        ):
            if not is_sha256_digest(digest):
                raise ValueError(f"lockfile {name} must be a tagged SHA-256 digest")
        if not isinstance(self.patch_hashes, Mapping):
            raise ValueError("lockfile patch_hashes must be an object")
        if len(self.patch_hashes) > MAX_STACK_LOCK_PATCHES:
            raise ValueError("lockfile patch count exceeds the limit")
        patch_hashes: dict[str, str] = {}
        for patch_id, manifest_hash in self.patch_hashes.items():
            if not is_sha256_digest(patch_id) or not is_sha256_digest(manifest_hash):
                raise ValueError(
                    "lockfile patch_hashes must map patch SHA-256 identities to "
                    "manifest SHA-256 digests"
                )
            patch_hashes[patch_id] = manifest_hash
        object.__setattr__(
            self,
            "patch_hashes",
            MappingProxyType(dict(sorted(patch_hashes.items()))),
        )
        if not isinstance(self.contract_hashes, tuple):
            raise ValueError("lockfile contract_hashes must be a tuple")
        if len(self.contract_hashes) > MAX_STACK_LOCK_CONTRACTS:
            raise ValueError("lockfile contract count exceeds the limit")
        if not all(is_sha256_digest(item) for item in self.contract_hashes):
            raise ValueError("lockfile contract_hashes must contain tagged SHA-256 digests")
        if tuple(sorted(set(self.contract_hashes))) != self.contract_hashes:
            raise ValueError("lockfile contract_hashes must be sorted and unique")
        if not patch_hashes and self.contract_hashes:
            raise ValueError("an empty stack cannot claim contracts")
        for name, optional_digest in (
            ("resolved_artifact_hash", self.resolved_artifact_hash),
            ("certificate_hash", self.certificate_hash),
            ("audit_hash", self.audit_hash),
        ):
            if optional_digest is not None and not is_sha256_digest(optional_digest):
                raise ValueError(f"lockfile {name} must be null or a tagged SHA-256 digest")
        successful = self.resolution in {
            StackResolutionKind.NAIVE_ADDITIVE_STACK,
            StackResolutionKind.VERIFIED_COMPOSITE_PATCH,
        }
        if successful and self.resolved_artifact_hash is None:
            raise ValueError("successful lockfile resolution must pin a resolved artifact")
        if successful and patch_hashes and self.certificate_hash is None:
            raise ValueError("successful nonempty stack must pin a verification certificate")
        if not successful and (
            self.resolved_artifact_hash is not None or self.certificate_hash is not None
        ):
            raise ValueError("unsuccessful lockfile resolution cannot pin a resolved artifact")
        if not patch_hashes and successful and self.resolved_artifact_hash != self.base_hash:
            raise ValueError("an empty successful stack must resolve to its pinned base")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "base_hash": self.base_hash,
            "patch_hashes": {
                patch_id: self.patch_hashes[patch_id] for patch_id in sorted(self.patch_hashes)
            },
            "contract_hashes": list(self.contract_hashes),
            "resolved_artifact_hash": self.resolved_artifact_hash,
            "verification_policy_hash": self.verification_policy_hash,
            "resolution": self.resolution.value,
            "certificate_hash": self.certificate_hash,
            "audit_hash": self.audit_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StackLock:
        """Strictly parse the data-only core of Patch Stack Lockfile v1."""

        validate_data_shape(value, limits=STACK_LOCK_LIMITS)
        unknown = set(value) - STACK_LOCK_FIELDS
        missing = STACK_LOCK_FIELDS - set(value)
        if unknown:
            raise ValueError(f"unknown Patch Stack Lockfile v1 fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing Patch Stack Lockfile v1 fields: {sorted(missing)}")
        if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
            raise ValueError("unsupported Patch Stack Lockfile schema version")

        def required_digest(name: str) -> str:
            candidate = value.get(name)
            if not is_sha256_digest(candidate):
                raise ValueError(f"lockfile {name} must be a tagged SHA-256 digest")
            return cast(str, candidate)

        def optional_digest(name: str) -> str | None:
            candidate = value.get(name)
            if candidate is not None and not is_sha256_digest(candidate):
                raise ValueError(f"lockfile {name} must be null or a tagged SHA-256 digest")
            return cast(str | None, candidate)

        raw_patch_hashes = value.get("patch_hashes")
        if not isinstance(raw_patch_hashes, Mapping):
            raise ValueError("lockfile patch_hashes must be an object")
        if len(raw_patch_hashes) > MAX_STACK_LOCK_PATCHES:
            raise ValueError("lockfile patch count exceeds the limit")
        patch_hashes: dict[str, str] = {}
        for patch_id, manifest_hash in raw_patch_hashes.items():
            if not is_sha256_digest(patch_id) or not is_sha256_digest(manifest_hash):
                raise ValueError(
                    "lockfile patch_hashes must map patch SHA-256 identities to "
                    "manifest SHA-256 digests"
                )
            patch_hashes[cast(str, patch_id)] = cast(str, manifest_hash)

        raw_contract_hashes = value.get("contract_hashes")
        if not isinstance(raw_contract_hashes, list):
            raise ValueError("lockfile contract_hashes must be an array")
        if len(raw_contract_hashes) > MAX_STACK_LOCK_CONTRACTS:
            raise ValueError("lockfile contract count exceeds the limit")
        if not all(is_sha256_digest(item) for item in raw_contract_hashes):
            raise ValueError("lockfile contract_hashes must contain tagged SHA-256 digests")
        contract_hashes = tuple(cast(list[str], raw_contract_hashes))
        if tuple(sorted(set(contract_hashes))) != contract_hashes:
            raise ValueError("lockfile contract_hashes must be sorted and unique")

        raw_resolution = value.get("resolution")
        if not isinstance(raw_resolution, str):
            raise ValueError("lockfile resolution must be a string")
        try:
            resolution = StackResolutionKind(raw_resolution)
        except ValueError as error:
            raise ValueError(f"unsupported lockfile resolution: {raw_resolution}") from error

        resolved_artifact_hash = optional_digest("resolved_artifact_hash")
        if (
            resolution
            in {
                StackResolutionKind.NAIVE_ADDITIVE_STACK,
                StackResolutionKind.VERIFIED_COMPOSITE_PATCH,
            }
            and resolved_artifact_hash is None
        ):
            raise ValueError("successful lockfile resolution must pin a resolved artifact")

        return cls(
            schema_version=1,
            base_hash=required_digest("base_hash"),
            patch_hashes=dict(sorted(patch_hashes.items())),
            contract_hashes=contract_hashes,
            resolved_artifact_hash=resolved_artifact_hash,
            verification_policy_hash=required_digest("verification_policy_hash"),
            resolution=resolution,
            certificate_hash=optional_digest("certificate_hash"),
            audit_hash=optional_digest("audit_hash"),
        )


def loads_stack_lock(text: str | bytes) -> StackLock:
    """Parse a canonical, bounded core Patch Stack Lockfile v1 record."""

    value = loads_data(
        text,
        format="json",
        limits=STACK_LOCK_LIMITS,
        require_canonical=True,
    )
    if not isinstance(value, Mapping):
        raise ValueError("Patch Stack Lockfile root must be an object")
    return StackLock.from_dict(cast(Mapping[str, object], value))


def read_stack_lock(path: str | Path) -> StackLock:
    """Read a regular non-symlink core Patch Stack Lockfile v1 record."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("stack lockfile must be a regular file")
    if source.stat().st_size > MAX_STACK_LOCK_BYTES:
        raise ValueError("stack lockfile exceeds size limit")
    return loads_stack_lock(source.read_bytes())


@dataclass(frozen=True, slots=True)
class ResolvedStack:
    request: StackResolutionRequest
    execution: StackResolutionExecution
    lock: StackLock
    dependency_order: tuple[str, ...]


def dependency_order(patches: tuple[PatchReference, ...] | list[PatchReference]) -> tuple[str, ...]:
    """Order patches by required/provided contract identities."""

    by_id = {patch.patch_id: patch for patch in patches}
    if len(by_id) != len(patches):
        raise ValueError("stack contains duplicate patch identities")
    providers: dict[str, list[str]] = {}
    for patch in patches:
        for contract_id in patch.provided_contracts:
            providers.setdefault(contract_id, []).append(patch.patch_id)
    for patch in patches:
        missing = sorted(set(patch.requires) - set(providers))
        if missing:
            raise ValueError(
                f"patch {patch.patch_id!r} has unsatisfied required contracts: {missing}"
            )
        ambiguous = sorted(
            contract_id
            for contract_id in patch.requires
            if len(set(providers[contract_id]) - {patch.patch_id}) > 1
            and patch.patch_id not in providers[contract_id]
        )
        if ambiguous:
            raise ValueError(
                f"patch {patch.patch_id!r} has ambiguously provided required contracts: {ambiguous}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(patch_id: str) -> None:
        if patch_id in visited:
            return
        if patch_id in visiting:
            raise ValueError(f"patch dependency cycle contains {patch_id!r}")
        visiting.add(patch_id)
        for contract_id in sorted(by_id[patch_id].requires):
            candidates = sorted(set(providers[contract_id]) - {patch_id})
            if candidates:
                visit(candidates[0])
        visiting.remove(patch_id)
        visited.add(patch_id)
        ordered.append(patch_id)

    for patch_id in sorted(by_id):
        visit(patch_id)
    return tuple(ordered)


def resolve_stack(
    *,
    base_hash: str,
    patches: tuple[PatchReference, ...] | list[PatchReference],
    resolver: StackResolver,
    repair_conflicts: bool,
    subset_audit_budget: int,
) -> ResolvedStack:
    if not base_hash:
        raise ValueError("base_hash must not be empty")
    if subset_audit_budget < 0:
        raise ValueError("subset_audit_budget must be non-negative")
    patch_tuple = tuple(patches)
    order = dependency_order(patch_tuple)
    by_id = {patch.patch_id: patch for patch in patch_tuple}
    wrong_bases = sorted(patch.patch_id for patch in patch_tuple if patch.base_hash != base_hash)
    if wrong_bases:
        raise ValueError(f"patches are incompatible with the declared base: {wrong_bases}")
    # Patch listing order has no additive meaning.  Use identity order in the
    # resolver request; dependency_order is retained as separately audited data.
    identity_ordered = tuple(by_id[patch_id] for patch_id in sorted(by_id))
    request = StackResolutionRequest(
        base_hash=base_hash,
        patches=identity_ordered,
        repair_conflicts=repair_conflicts,
        subset_audit_budget=subset_audit_budget,
    )
    execution = resolver(request)
    all_contract_hashes = tuple(
        sorted(
            {contract_hash for patch in identity_ordered for contract_hash in patch.contract_hashes}
        )
    )
    lock = StackLock(
        schema_version=1,
        base_hash=base_hash,
        patch_hashes={patch.patch_id: patch.patch_hash for patch in identity_ordered},
        contract_hashes=all_contract_hashes,
        resolved_artifact_hash=execution.resolved_artifact_hash,
        verification_policy_hash=execution.verification_policy_hash,
        resolution=execution.kind,
        certificate_hash=execution.certificate_hash,
        audit_hash=execution.audit_hash,
    )
    return ResolvedStack(request=request, execution=execution, lock=lock, dependency_order=order)
