"""Semantic-merge orchestration around a supplied joint patch compiler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import torch

from modelpact.compose.closure import (
    CompositionExecutor,
    CompositionResult,
    PatchOperand,
    VerificationReport,
    verify_contract_closure,
)
from modelpact.compose.contradiction import ContradictionChecker
from modelpact.status import CompositionClaim, VerificationOutcome


class MergeDisposition(StrEnum):
    NAIVE_COMPOSITION_RETURNED = "NAIVE_COMPOSITION_RETURNED"
    SEMANTIC_MERGE_VERIFIED = "SEMANTIC_MERGE_VERIFIED"
    STATIC_CONTRACT_CONTRADICTION = "STATIC_CONTRACT_CONTRADICTION"
    STRUCTURAL_INCOMPATIBILITY = "STRUCTURAL_INCOMPATIBILITY"
    EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET = "EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET"
    COMPILER_FAILED = "COMPILER_FAILED"
    RECOMPILED_CANDIDATE_FAILED_VERIFICATION = "RECOMPILED_CANDIDATE_FAILED_VERIFICATION"
    FINAL_CANDIDATE_FAILED_HOLDOUT = "FINAL_CANDIDATE_FAILED_HOLDOUT"


@dataclass(frozen=True, slots=True)
class MergeBudget:
    maximum_steps: int
    maximum_restarts: int = 1
    maximum_trainable_parameters: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_steps <= 0 or self.maximum_restarts <= 0:
            raise ValueError("merge optimization steps and restarts must be positive")
        if self.maximum_trainable_parameters is not None and self.maximum_trainable_parameters <= 0:
            raise ValueError("maximum_trainable_parameters must be positive when declared")


@dataclass(frozen=True, slots=True)
class SemanticMergeRequest:
    parent_patch_ids: tuple[str, ...]
    base_signature: str
    module_schema_hash: str
    contract_ids: tuple[str, ...]
    initial_delta: Mapping[str, torch.Tensor]
    parent_deltas: Mapping[str, Mapping[str, torch.Tensor]]
    budget: MergeBudget


@dataclass(frozen=True, slots=True)
class JointCompilationResult:
    candidate_delta: Mapping[str, torch.Tensor] | None
    optimization_succeeded: bool
    budget_exhausted: bool
    steps_executed: int
    restarts_executed: int
    best_margins: Mapping[str, float] = field(default_factory=dict)
    violated_contracts: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.steps_executed < 0 or self.restarts_executed < 0:
            raise ValueError("executed optimization counts must be non-negative")
        if self.optimization_succeeded and self.candidate_delta is None:
            raise ValueError("successful joint compilation must return a candidate delta")


class JointCompiler(Protocol):
    def __call__(self, request: SemanticMergeRequest) -> JointCompilationResult: ...


@dataclass(frozen=True, slots=True)
class SemanticMergeResult:
    disposition: MergeDisposition
    claim: CompositionClaim
    parent_patch_ids: tuple[str, ...]
    contract_ids: tuple[str, ...]
    compiler_invoked: bool
    delta: Mapping[str, torch.Tensor]
    naive_composition: CompositionResult
    verification: VerificationReport | None = None
    compilation: JointCompilationResult | None = None
    warnings: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.disposition in {
            MergeDisposition.NAIVE_COMPOSITION_RETURNED,
            MergeDisposition.SEMANTIC_MERGE_VERIFIED,
        }


def semantic_merge(
    operands: tuple[PatchOperand, ...] | list[PatchOperand],
    *,
    executor: CompositionExecutor,
    compiler: JointCompiler,
    budget: MergeBudget,
    aliases: Mapping[str, str] | None = None,
    contradiction_checker: ContradictionChecker | None = None,
    force_recompile: bool = False,
    execute_baselines: bool = False,
) -> SemanticMergeResult:
    """Verify addition, then jointly compile and independently verify if needed."""

    parents = tuple(operands)
    naive = verify_contract_closure(
        parents,
        executor=executor,
        aliases=aliases,
        contradiction_checker=contradiction_checker,
        execute_baselines=execute_baselines,
    )
    if naive.claim is CompositionClaim.STATIC_CONTRACT_CONTRADICTION:
        return SemanticMergeResult(
            disposition=MergeDisposition.STATIC_CONTRACT_CONTRADICTION,
            claim=naive.claim,
            parent_patch_ids=naive.patch_ids,
            contract_ids=naive.contract_ids,
            compiler_invoked=False,
            delta={},
            naive_composition=naive,
        )
    if naive.claim is CompositionClaim.STRUCTURAL_INCOMPATIBILITY:
        return SemanticMergeResult(
            disposition=MergeDisposition.STRUCTURAL_INCOMPATIBILITY,
            claim=naive.claim,
            parent_patch_ids=naive.patch_ids,
            contract_ids=naive.contract_ids,
            compiler_invoked=False,
            delta={},
            naive_composition=naive,
        )
    if naive.claim is CompositionClaim.COMPOSITION_CLOSED and not force_recompile:
        return SemanticMergeResult(
            disposition=MergeDisposition.NAIVE_COMPOSITION_RETURNED,
            claim=CompositionClaim.COMPOSITION_CLOSED,
            parent_patch_ids=naive.patch_ids,
            contract_ids=naive.contract_ids,
            compiler_invoked=False,
            delta=naive.resolved_delta,
            naive_composition=naive,
            verification=naive.verification,
        )

    ordered = tuple(sorted(parents, key=lambda item: item.patch_id))
    request = SemanticMergeRequest(
        parent_patch_ids=tuple(item.patch_id for item in ordered),
        base_signature=ordered[0].base_signature,
        module_schema_hash=ordered[0].module_schema_hash,
        contract_ids=naive.contract_ids,
        initial_delta=naive.resolved_delta,
        parent_deltas={item.patch_id: item.delta for item in ordered},
        budget=budget,
    )
    compilation = compiler(request)
    if not compilation.optimization_succeeded or compilation.candidate_delta is None:
        disposition = (
            MergeDisposition.EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET
            if compilation.budget_exhausted
            else MergeDisposition.COMPILER_FAILED
        )
        claim = (
            CompositionClaim.EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET
            if compilation.budget_exhausted
            else CompositionClaim.SEMANTIC_CONFLICT
        )
        return SemanticMergeResult(
            disposition=disposition,
            claim=claim,
            parent_patch_ids=naive.patch_ids,
            contract_ids=naive.contract_ids,
            compiler_invoked=True,
            delta=compilation.candidate_delta or {},
            naive_composition=naive,
            compilation=compilation,
        )

    verification = executor(compilation.candidate_delta, naive.contract_ids)
    reported = set(verification.by_contract())
    verified = (
        verification.outcome is VerificationOutcome.PASS
        and set(naive.contract_ids) <= reported
        and all(margin.passed for margin in verification.margins)
    )
    if not verified:
        return SemanticMergeResult(
            disposition=MergeDisposition.RECOMPILED_CANDIDATE_FAILED_VERIFICATION,
            claim=CompositionClaim.SEMANTIC_CONFLICT,
            parent_patch_ids=naive.patch_ids,
            contract_ids=naive.contract_ids,
            compiler_invoked=True,
            delta=compilation.candidate_delta,
            naive_composition=naive,
            verification=verification,
            compilation=compilation,
            warnings=("joint compiler returned a candidate that failed independent verification",),
        )
    return SemanticMergeResult(
        disposition=MergeDisposition.SEMANTIC_MERGE_VERIFIED,
        claim=CompositionClaim.COMPOSITION_CLOSED,
        parent_patch_ids=naive.patch_ids,
        contract_ids=naive.contract_ids,
        compiler_invoked=True,
        delta=compilation.candidate_delta,
        naive_composition=naive,
        verification=verification,
        compilation=compilation,
    )
