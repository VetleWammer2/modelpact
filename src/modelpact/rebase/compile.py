"""Direct-first semantic rebase orchestration with independent verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import torch

from modelpact.rebase.direct import (
    BaseModelDescriptor,
    DirectPatchApplier,
    DirectTransferResult,
    RebaseCompatibility,
    RebasePatch,
    RebaseVerification,
    RebaseVerifier,
    assess_compatibility,
    attempt_direct_transfer,
)
from modelpact.rebase.evidence import RebaseEvidence
from modelpact.status import RebaseClaim


class RebaseDisposition(StrEnum):
    DIRECT_TRANSPLANT_VERIFIED = "DIRECT_TRANSPLANT_VERIFIED"
    SEMANTIC_REBASE_VERIFIED = "SEMANTIC_REBASE_VERIFIED"
    SOURCE_BASE_MISMATCH = "SOURCE_BASE_MISMATCH"
    INCOMPATIBLE_SEMANTICS = "INCOMPATIBLE_SEMANTICS"
    CROSS_ARCHITECTURE_DISABLED = "CROSS_ARCHITECTURE_DISABLED"
    INSUFFICIENT_TEACHER_EVIDENCE = "INSUFFICIENT_TEACHER_EVIDENCE"
    RECOMPILE_FAILED = "RECOMPILE_FAILED"
    RECOMPILED_CANDIDATE_FAILED_VERIFICATION = "RECOMPILED_CANDIDATE_FAILED_VERIFICATION"


@dataclass(frozen=True, slots=True)
class RebaseBudget:
    maximum_steps: int
    maximum_restarts: int = 1

    def __post_init__(self) -> None:
        if self.maximum_steps <= 0 or self.maximum_restarts <= 0:
            raise ValueError("rebase optimization steps and restarts must be positive")


@dataclass(frozen=True, slots=True)
class RebaseRequest:
    patch: RebasePatch
    source_base: BaseModelDescriptor
    target_base: BaseModelDescriptor
    new_base_guard_ids: tuple[str, ...]
    budget: RebaseBudget
    allow_cross_architecture: bool = True
    compiler_configuration: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TeacherContext:
    old_patched_teacher: object
    new_unpatched_teacher: object
    old_behavior_margins: Mapping[str, float]
    evidence_count: int

    def __post_init__(self) -> None:
        if self.evidence_count < 0:
            raise ValueError("teacher evidence count must be non-negative")


class TeacherBuilder(Protocol):
    def __call__(self, request: RebaseRequest) -> TeacherContext: ...


@dataclass(frozen=True, slots=True)
class BehavioralRecompileRequest:
    source_patch_id: str
    source_base: BaseModelDescriptor
    target_base: BaseModelDescriptor
    old_patched_teacher: object
    new_unpatched_teacher: object
    target_contract_ids: tuple[str, ...]
    guard_contract_ids: tuple[str, ...]
    budget: RebaseBudget
    compiler_configuration: Mapping[str, object]
    direct_transfer: DirectTransferResult


@dataclass(frozen=True, slots=True)
class BehavioralRecompileResult:
    candidate_delta: Mapping[str, torch.Tensor] | None
    optimization_succeeded: bool
    budget_exhausted: bool
    steps_executed: int
    restarts_executed: int
    best_target_margins: Mapping[str, float] = field(default_factory=dict)
    best_guard_margins: Mapping[str, float] = field(default_factory=dict)
    violated_contracts: tuple[str, ...] = ()
    complexity: Mapping[str, int | float] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.steps_executed < 0 or self.restarts_executed < 0:
            raise ValueError("executed optimization counts must be non-negative")
        if self.optimization_succeeded and self.candidate_delta is None:
            raise ValueError("successful behavioral recompilation must return a candidate")


class BehavioralRecompiler(Protocol):
    def __call__(self, request: BehavioralRecompileRequest) -> BehavioralRecompileResult: ...


@dataclass(frozen=True, slots=True)
class RebaseResult:
    disposition: RebaseDisposition
    claim: RebaseClaim
    delta: Mapping[str, torch.Tensor]
    direct_transfer: DirectTransferResult
    recompile: BehavioralRecompileResult | None
    verification: RebaseVerification | None
    evidence: RebaseEvidence
    rebased_from: str

    @property
    def verified(self) -> bool:
        return self.claim in {
            RebaseClaim.DIRECT_TRANSPLANT_VERIFIED,
            RebaseClaim.SEMANTIC_REBASE_VERIFIED,
        }


def _evidence(
    *,
    request: RebaseRequest,
    claim: RebaseClaim,
    direct: DirectTransferResult,
    recompile: BehavioralRecompileResult | None,
    verification: RebaseVerification | None,
    teacher: TeacherContext | None,
    warnings: tuple[str, ...] = (),
) -> RebaseEvidence:
    complexity_before: dict[str, int | float] = {
        "target_tensors": len(request.patch.delta),
        "parameters": sum(tensor.numel() for tensor in request.patch.delta.values()),
    }
    return RebaseEvidence(
        source_patch_id=request.patch.patch_id,
        source_base_hash=request.source_base.signature,
        target_base_hash=request.target_base.signature,
        claim=claim,
        compatibility=direct.compatibility.value,
        direct_attempted=direct.attempted,
        direct_outcome=(
            direct.verification.outcome.value if direct.verification is not None else None
        ),
        recompile_attempted=recompile is not None,
        recompile_steps=recompile.steps_executed if recompile else 0,
        recompile_restarts=recompile.restarts_executed if recompile else 0,
        budget_exhausted=recompile.budget_exhausted if recompile else False,
        old_patched_behavior=teacher.old_behavior_margins if teacher else {},
        new_patched_behavior=verification.target_margins if verification else {},
        new_base_preservation=verification.guard_margins if verification else {},
        patch_complexity_before=complexity_before,
        patch_complexity_after=(
            recompile.complexity
            if recompile
            else complexity_before
            if claim is RebaseClaim.DIRECT_TRANSPLANT_VERIFIED
            else {}
        ),
        warnings=warnings,
    )


def semantic_rebase(
    request: RebaseRequest,
    *,
    applier: DirectPatchApplier,
    verifier: RebaseVerifier,
    teacher_builder: TeacherBuilder,
    recompiler: BehavioralRecompiler,
) -> RebaseResult:
    """Verify direct transplant, otherwise compile behavior on the target base."""

    compatibility = assess_compatibility(request.source_base, request.target_base)
    direct = attempt_direct_transfer(
        request.patch,
        source=request.source_base,
        target=request.target_base,
        new_base_guard_ids=request.new_base_guard_ids,
        applier=applier,
        verifier=verifier,
    )
    if request.patch.source_base_signature != request.source_base.signature:
        claim = RebaseClaim.REBASE_INCONCLUSIVE
        evidence = _evidence(
            request=request,
            claim=claim,
            direct=direct,
            recompile=None,
            verification=None,
            teacher=None,
            warnings=("source patch identity could not be validated",),
        )
        return RebaseResult(
            disposition=RebaseDisposition.SOURCE_BASE_MISMATCH,
            claim=claim,
            delta={},
            direct_transfer=direct,
            recompile=None,
            verification=None,
            evidence=evidence,
            rebased_from=request.patch.patch_id,
        )
    if direct.verified:
        claim = RebaseClaim.DIRECT_TRANSPLANT_VERIFIED
        evidence = _evidence(
            request=request,
            claim=claim,
            direct=direct,
            recompile=None,
            verification=direct.verification,
            teacher=None,
        )
        return RebaseResult(
            disposition=RebaseDisposition.DIRECT_TRANSPLANT_VERIFIED,
            claim=claim,
            delta=request.patch.delta,
            direct_transfer=direct,
            recompile=None,
            verification=direct.verification,
            evidence=evidence,
            rebased_from=request.patch.patch_id,
        )
    if compatibility in {
        RebaseCompatibility.INCOMPATIBLE_TOKENIZER,
        RebaseCompatibility.INCOMPATIBLE_OUTPUT_SEMANTICS,
    }:
        claim = RebaseClaim.REBASE_INCONCLUSIVE
        evidence = _evidence(
            request=request,
            claim=claim,
            direct=direct,
            recompile=None,
            verification=None,
            teacher=None,
            warnings=("input/output semantics are not comparable for behavioral recompile",),
        )
        return RebaseResult(
            disposition=RebaseDisposition.INCOMPATIBLE_SEMANTICS,
            claim=claim,
            delta={},
            direct_transfer=direct,
            recompile=None,
            verification=None,
            evidence=evidence,
            rebased_from=request.patch.patch_id,
        )
    cross_architecture = request.source_base.architecture_id != request.target_base.architecture_id
    if cross_architecture and not request.allow_cross_architecture:
        claim = RebaseClaim.REBASE_INCONCLUSIVE
        evidence = _evidence(
            request=request,
            claim=claim,
            direct=direct,
            recompile=None,
            verification=None,
            teacher=None,
            warnings=("cross-architecture behavioral recompilation was disabled",),
        )
        return RebaseResult(
            disposition=RebaseDisposition.CROSS_ARCHITECTURE_DISABLED,
            claim=claim,
            delta={},
            direct_transfer=direct,
            recompile=None,
            verification=None,
            evidence=evidence,
            rebased_from=request.patch.patch_id,
        )

    teacher = teacher_builder(request)
    if teacher.evidence_count <= 0:
        claim = RebaseClaim.REBASE_INCONCLUSIVE
        evidence = _evidence(
            request=request,
            claim=claim,
            direct=direct,
            recompile=None,
            verification=None,
            teacher=teacher,
            warnings=("no target-domain teacher evidence was available",),
        )
        return RebaseResult(
            disposition=RebaseDisposition.INSUFFICIENT_TEACHER_EVIDENCE,
            claim=claim,
            delta={},
            direct_transfer=direct,
            recompile=None,
            verification=None,
            evidence=evidence,
            rebased_from=request.patch.patch_id,
        )
    targets = tuple(sorted(request.patch.target_contract_ids))
    guards = tuple(
        sorted(set(request.patch.preservation_contract_ids) | set(request.new_base_guard_ids))
    )
    compile_request = BehavioralRecompileRequest(
        source_patch_id=request.patch.patch_id,
        source_base=request.source_base,
        target_base=request.target_base,
        old_patched_teacher=teacher.old_patched_teacher,
        new_unpatched_teacher=teacher.new_unpatched_teacher,
        target_contract_ids=targets,
        guard_contract_ids=guards,
        budget=request.budget,
        compiler_configuration=request.compiler_configuration,
        direct_transfer=direct,
    )
    compilation = recompiler(compile_request)
    if not compilation.optimization_succeeded or compilation.candidate_delta is None:
        claim = RebaseClaim.REBASE_FAILED
        evidence = _evidence(
            request=request,
            claim=claim,
            direct=direct,
            recompile=compilation,
            verification=None,
            teacher=teacher,
            warnings=((compilation.failure_reason,) if compilation.failure_reason else ()),
        )
        return RebaseResult(
            disposition=RebaseDisposition.RECOMPILE_FAILED,
            claim=claim,
            delta=compilation.candidate_delta or {},
            direct_transfer=direct,
            recompile=compilation,
            verification=None,
            evidence=evidence,
            rebased_from=request.patch.patch_id,
        )
    candidate = applier(compilation.candidate_delta, request.target_base)
    verification = verifier(candidate, targets, guards)
    if not verification.passes(targets, guards):
        claim = RebaseClaim.REBASE_FAILED
        evidence = _evidence(
            request=request,
            claim=claim,
            direct=direct,
            recompile=compilation,
            verification=verification,
            teacher=teacher,
            warnings=("recompiled candidate failed independent behavioral verification",),
        )
        return RebaseResult(
            disposition=RebaseDisposition.RECOMPILED_CANDIDATE_FAILED_VERIFICATION,
            claim=claim,
            delta=compilation.candidate_delta,
            direct_transfer=direct,
            recompile=compilation,
            verification=verification,
            evidence=evidence,
            rebased_from=request.patch.patch_id,
        )
    claim = RebaseClaim.SEMANTIC_REBASE_VERIFIED
    evidence = _evidence(
        request=request,
        claim=claim,
        direct=direct,
        recompile=compilation,
        verification=verification,
        teacher=teacher,
    )
    return RebaseResult(
        disposition=RebaseDisposition.SEMANTIC_REBASE_VERIFIED,
        claim=claim,
        delta=compilation.candidate_delta,
        direct_transfer=direct,
        recompile=compilation,
        verification=verification,
        evidence=evidence,
        rebased_from=request.patch.patch_id,
    )
