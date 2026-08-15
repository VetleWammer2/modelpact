"""Deterministic lineage and declarative patch-stack resolution records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


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
        if self.patch_id in self.requires:
            raise ValueError("a patch cannot depend on itself")


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


@dataclass(frozen=True, slots=True)
class ResolvedStack:
    request: StackResolutionRequest
    execution: StackResolutionExecution
    lock: StackLock
    dependency_order: tuple[str, ...]


def dependency_order(patches: tuple[PatchReference, ...] | list[PatchReference]) -> tuple[str, ...]:
    """Return a stable topological order, rejecting missing dependencies/cycles."""

    by_id = {patch.patch_id: patch for patch in patches}
    if len(by_id) != len(patches):
        raise ValueError("stack contains duplicate patch identities")
    for patch in patches:
        missing = sorted(set(patch.requires) - set(by_id))
        if missing:
            raise ValueError(f"patch {patch.patch_id!r} has missing dependencies: {missing}")

    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(patch_id: str) -> None:
        if patch_id in visited:
            return
        if patch_id in visiting:
            raise ValueError(f"patch dependency cycle contains {patch_id!r}")
        visiting.add(patch_id)
        for dependency in sorted(by_id[patch_id].requires):
            visit(dependency)
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
