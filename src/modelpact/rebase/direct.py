"""Structural compatibility assessment and verified direct patch transfer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch

from modelpact.status import VerificationOutcome


@dataclass(frozen=True, slots=True)
class BaseModelDescriptor:
    signature: str
    architecture_id: str
    module_schema_hash: str
    tokenizer_hash: str
    output_semantics: str
    module_shapes: Mapping[str, tuple[int, ...]]
    family_id: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.signature,
            self.architecture_id,
            self.module_schema_hash,
            self.tokenizer_hash,
            self.output_semantics,
        )
        if any(not value for value in required):
            raise ValueError("model compatibility identities must not be empty")
        if any(any(dimension <= 0 for dimension in shape) for shape in self.module_shapes.values()):
            raise ValueError("module shapes must contain positive dimensions")


@dataclass(frozen=True, slots=True)
class RebasePatch:
    patch_id: str
    source_base_signature: str
    delta: Mapping[str, torch.Tensor]
    target_contract_ids: tuple[str, ...]
    preservation_contract_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.patch_id or not self.source_base_signature:
            raise ValueError("patch and source-base identities must not be empty")
        if not self.target_contract_ids or not self.preservation_contract_ids:
            raise ValueError("rebased patches require target and preservation contracts")
        contract_ids = (*self.target_contract_ids, *self.preservation_contract_ids)
        if len(contract_ids) != len(set(contract_ids)):
            raise ValueError("target and preservation contract identities must be unique")


class RebaseCompatibility(StrEnum):
    DIRECT_PHYSICAL_TRANSFER = "DIRECT_PHYSICAL_TRANSFER"
    BEHAVIORAL_RECOMPILE_ONLY = "BEHAVIORAL_RECOMPILE_ONLY"
    INCOMPATIBLE_TOKENIZER = "INCOMPATIBLE_TOKENIZER"
    INCOMPATIBLE_OUTPUT_SEMANTICS = "INCOMPATIBLE_OUTPUT_SEMANTICS"


def assess_compatibility(
    source: BaseModelDescriptor, target: BaseModelDescriptor
) -> RebaseCompatibility:
    if source.output_semantics != target.output_semantics:
        return RebaseCompatibility.INCOMPATIBLE_OUTPUT_SEMANTICS
    if source.tokenizer_hash != target.tokenizer_hash:
        return RebaseCompatibility.INCOMPATIBLE_TOKENIZER
    same_physical_schema = (
        source.architecture_id == target.architecture_id
        and source.module_schema_hash == target.module_schema_hash
        and dict(source.module_shapes) == dict(target.module_shapes)
    )
    if same_physical_schema:
        return RebaseCompatibility.DIRECT_PHYSICAL_TRANSFER
    return RebaseCompatibility.BEHAVIORAL_RECOMPILE_ONLY


@dataclass(frozen=True, slots=True)
class RebaseVerification:
    outcome: VerificationOutcome
    target_margins: Mapping[str, float]
    guard_margins: Mapping[str, float]
    prompt_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        margins = (*self.target_margins.values(), *self.guard_margins.values())
        if not all(math.isfinite(value) for value in margins):
            raise ValueError("rebase verification margins must be finite")
        if self.outcome is VerificationOutcome.PASS and any(value < 0.0 for value in margins):
            raise ValueError("PASS rebase verification cannot contain a negative margin")

    def covers(self, targets: tuple[str, ...], guards: tuple[str, ...]) -> bool:
        return set(targets) <= set(self.target_margins) and set(guards) <= set(self.guard_margins)

    def passes(self, targets: tuple[str, ...], guards: tuple[str, ...]) -> bool:
        return (
            self.outcome is VerificationOutcome.PASS
            and self.covers(targets, guards)
            and all(self.target_margins[contract] >= 0.0 for contract in targets)
            and all(self.guard_margins[contract] >= 0.0 for contract in guards)
        )


class DirectPatchApplier(Protocol):
    def __call__(
        self, delta: Mapping[str, torch.Tensor], target: BaseModelDescriptor
    ) -> object: ...


class RebaseVerifier(Protocol):
    def __call__(
        self,
        candidate: object,
        target_contract_ids: tuple[str, ...],
        guard_contract_ids: tuple[str, ...],
    ) -> RebaseVerification: ...


@dataclass(frozen=True, slots=True)
class DirectTransferResult:
    compatibility: RebaseCompatibility
    attempted: bool
    verification: RebaseVerification | None
    verified: bool
    reason: str | None = None


def attempt_direct_transfer(
    patch: RebasePatch,
    *,
    source: BaseModelDescriptor,
    target: BaseModelDescriptor,
    new_base_guard_ids: tuple[str, ...],
    applier: DirectPatchApplier,
    verifier: RebaseVerifier,
) -> DirectTransferResult:
    if patch.source_base_signature != source.signature:
        return DirectTransferResult(
            compatibility=assess_compatibility(source, target),
            attempted=False,
            verification=None,
            verified=False,
            reason="patch source-base signature does not match the declared source base",
        )
    compatibility = assess_compatibility(source, target)
    if compatibility is not RebaseCompatibility.DIRECT_PHYSICAL_TRANSFER:
        return DirectTransferResult(
            compatibility=compatibility,
            attempted=False,
            verification=None,
            verified=False,
            reason="physical tensor transfer is not structurally compatible",
        )
    candidate = applier(patch.delta, target)
    guards = tuple(sorted(set(patch.preservation_contract_ids) | set(new_base_guard_ids)))
    verification = verifier(candidate, tuple(sorted(patch.target_contract_ids)), guards)
    verified = verification.passes(tuple(sorted(patch.target_contract_ids)), guards)
    return DirectTransferResult(
        compatibility=compatibility,
        attempted=True,
        verification=verification,
        verified=verified,
        reason=None if verified else "directly transplanted delta failed behavioral verification",
    )
