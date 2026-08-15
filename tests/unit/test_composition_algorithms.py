from __future__ import annotations

import math

import pytest
import torch

from modelpact.compose.closure import (
    ContractMargin,
    MarginKind,
    PatchOperand,
    VerificationReport,
    additive_compose,
    verify_contract_closure,
)
from modelpact.compose.contradiction import (
    StaticRequirement,
    find_static_contradictions,
)
from modelpact.compose.interactions import (
    contract_margin_interaction,
    low_rank_subspace_diagnostics,
    module_overlap,
)
from modelpact.compose.merge import (
    JointCompilationResult,
    MergeBudget,
    MergeDisposition,
    semantic_merge,
)
from modelpact.compose.stack import (
    PatchReference,
    StackResolutionExecution,
    StackResolutionKind,
    dependency_order,
    resolve_stack,
)
from modelpact.status import CompositionClaim, VerificationOutcome


def _patch(
    patch_id: str,
    value: float,
    contract_id: str,
    *,
    margins: dict[str, float] | None = None,
) -> PatchOperand:
    return PatchOperand(
        patch_id=patch_id,
        base_signature="base",
        module_schema_hash="schema",
        delta={"weight": torch.tensor([value])},
        contract_ids=(contract_id,),
        verified_margins=margins or {contract_id: 1.0},
    )


def test_additive_composition_is_order_independent_and_alias_safe() -> None:
    first = PatchOperand(
        patch_id="a",
        base_signature="base",
        module_schema_hash="schema",
        delta={
            "embedding.weight": torch.tensor([1.0, 2.0]),
            "lm_head.weight": torch.tensor([1.0, 2.0]),
        },
        contract_ids=("a-contract",),
    )
    second = PatchOperand(
        patch_id="b",
        base_signature="base",
        module_schema_hash="schema",
        delta={"embedding.weight": torch.tensor([3.0, 4.0])},
        contract_ids=("b-contract",),
    )
    aliases = {"lm_head.weight": "embedding.weight"}
    forward = additive_compose([first, second], aliases=aliases)
    reverse = additive_compose([second, first], aliases=aliases)
    assert tuple(forward) == ("embedding.weight",)
    assert torch.equal(forward["embedding.weight"], torch.tensor([4.0, 6.0]))
    assert torch.equal(forward["embedding.weight"], reverse["embedding.weight"])


def test_contract_closure_executes_union_and_computes_semantic_interaction() -> None:
    left = _patch("left", 1.0, "left-contract", margins={"left-contract": 0.8})
    right = _patch("right", 2.0, "right-contract", margins={"right-contract": 0.7})
    observed_contracts: tuple[str, ...] = ()

    def execute(
        delta: dict[str, torch.Tensor] | object, contract_ids: tuple[str, ...]
    ) -> VerificationReport:
        nonlocal observed_contracts
        observed_contracts = contract_ids
        assert isinstance(delta, dict)
        assert torch.equal(delta["weight"], torch.tensor([3.0]))
        return VerificationReport(
            VerificationOutcome.PASS,
            (
                ContractMargin("left-contract", MarginKind.TARGET, 0.4),
                ContractMargin("right-contract", MarginKind.GUARD, 0.5),
            ),
        )

    result = verify_contract_closure([right, left], executor=execute)
    assert result.claim is CompositionClaim.COMPOSITION_CLOSED
    assert observed_contracts == ("left-contract", "right-contract")
    assert not result.unverified_contracts
    assert contract_margin_interaction(
        base_margin=0.1,
        left_margin=0.6,
        right_margin=0.5,
        composed_margin=0.2,
    ) == pytest.approx(-0.8)


def test_missing_or_failed_union_contract_is_semantic_conflict() -> None:
    result = verify_contract_closure(
        [_patch("a", 1.0, "a"), _patch("b", 1.0, "b")],
        executor=lambda _delta, _contracts: VerificationReport(
            VerificationOutcome.PASS,
            (ContractMargin("a", MarginKind.TARGET, 0.1),),
        ),
    )
    assert result.claim is CompositionClaim.SEMANTIC_CONFLICT
    assert result.unverified_contracts == ("b",)


def test_passing_margin_degradation_is_not_reported_as_closed() -> None:
    patch = _patch("a", 1.0, "a", margins={"a": 1.0})
    result = verify_contract_closure(
        [patch],
        executor=lambda _delta, _contracts: VerificationReport(
            VerificationOutcome.PASS,
            (ContractMargin("a", MarginKind.TARGET, 0.2),),
        ),
        degradation_tolerance=0.5,
    )
    assert result.claim is CompositionClaim.COMPOSITION_DEGRADED
    assert result.degraded_contracts == ("a",)


def test_static_checker_is_conservative_and_returns_minimal_witnesses() -> None:
    contradictions = find_static_contradictions(
        [
            StaticRequirement("a", "prompt-hash", "exact_match", "equals", "yes"),
            StaticRequirement("b", "prompt-hash", "exact_match", "equals", "no"),
            StaticRequirement("c", "other", "margin", ">=", 0.2),
            StaticRequirement("d", "other", "margin", "<=", 0.1),
        ]
    )
    assert {witness.code for witness in contradictions} == {
        "INCOMPATIBLE_EXACT_REQUIREMENTS",
        "EMPTY_NUMERIC_INTERVAL",
    }
    assert find_static_contradictions(
        [StaticRequirement("a", "x", "regex", "unknown", "a.*")]
    ) == ()


def test_subspace_diagnostics_report_orthogonal_and_shared_spaces() -> None:
    diagnostics = low_rank_subspace_diagnostics(
        left_left_factor=torch.tensor([[1.0], [0.0]]),
        left_right_factor=torch.tensor([[1.0, 0.0]]),
        right_left_factor=torch.tensor([[0.0], [1.0]]),
        right_right_factor=torch.tensor([[1.0, 0.0]]),
    )
    assert diagnostics.column_space.radians[0] == math.pi / 2
    assert diagnostics.row_space.radians[0] == 0.0
    assert module_overlap(["a", "b"], ["b", "c"]).jaccard == 1 / 3


def test_semantic_merge_invokes_joint_compiler_and_reverifies() -> None:
    calls = {"compiler": 0, "executor": 0}

    def execute(
        delta: dict[str, torch.Tensor] | object, contracts: tuple[str, ...]
    ) -> VerificationReport:
        calls["executor"] += 1
        assert isinstance(delta, dict)
        passing = float(delta["weight"].item()) < 1.5
        outcome = VerificationOutcome.PASS if passing else VerificationOutcome.FAIL
        margin = 0.5 if passing else -0.5
        return VerificationReport(
            outcome,
            tuple(ContractMargin(contract, MarginKind.TARGET, margin) for contract in contracts),
        )

    def compile_joint(request: object) -> JointCompilationResult:
        calls["compiler"] += 1
        assert request.parent_patch_ids == ("a", "b")
        return JointCompilationResult(
            candidate_delta={"weight": torch.tensor([0.75])},
            optimization_succeeded=True,
            budget_exhausted=False,
            steps_executed=8,
            restarts_executed=1,
        )

    result = semantic_merge(
        [_patch("a", 1.0, "a"), _patch("b", 1.0, "b")],
        executor=execute,
        compiler=compile_joint,
        budget=MergeBudget(maximum_steps=10),
    )
    assert result.disposition is MergeDisposition.SEMANTIC_MERGE_VERIFIED
    assert result.compiler_invoked
    assert calls == {"compiler": 1, "executor": 2}


def test_semantic_merge_reports_budgeted_empirical_infeasibility_honestly() -> None:
    failed = VerificationReport(
        VerificationOutcome.FAIL,
        (ContractMargin("a", MarginKind.TARGET, -1.0),),
    )
    result = semantic_merge(
        [_patch("a", 1.0, "a")],
        executor=lambda _delta, _contracts: failed,
        compiler=lambda _request: JointCompilationResult(
            candidate_delta=None,
            optimization_succeeded=False,
            budget_exhausted=True,
            steps_executed=12,
            restarts_executed=2,
            best_margins={"a": -0.1},
            violated_contracts=("a",),
        ),
        budget=MergeBudget(maximum_steps=12, maximum_restarts=2),
        force_recompile=True,
    )
    assert result.claim is CompositionClaim.EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET
    assert result.compilation is not None
    assert result.compilation.best_margins == {"a": -0.1}


def test_stack_resolution_is_identity_ordered_and_dependency_checked() -> None:
    base = "sha256:base"
    first = PatchReference("a", "ha", base, ("ca",), "aa")
    second = PatchReference("b", "hb", base, ("cb",), "ab", requires=("a",))
    assert dependency_order([second, first]) == ("a", "b")

    resolved = resolve_stack(
        base_hash=base,
        patches=[second, first],
        resolver=lambda request: StackResolutionExecution(
            StackResolutionKind.NAIVE_ADDITIVE_STACK,
            "resolved",
            "policy",
            "union",
        ),
        repair_conflicts=False,
        subset_audit_budget=8,
    )
    assert tuple(patch.patch_id for patch in resolved.request.patches) == ("a", "b")
    assert resolved.lock.to_dict()["patch_hashes"] == {"a": "ha", "b": "hb"}
