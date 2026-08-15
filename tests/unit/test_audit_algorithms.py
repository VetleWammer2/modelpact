from __future__ import annotations

from itertools import product

import pytest

from modelpact.audit.active import AuditConfig, SubsetEvaluation, audit_patch_pool
from modelpact.audit.reduce import ddmin_failing_subset
from modelpact.audit.subsets import enumerate_subsets
from modelpact.audit.surrogate import (
    PseudoBooleanRegressor,
    fit_bootstrap_ensemble,
    hierarchical_feature_terms,
)
from modelpact.status import AuditClaim, CompositionCoverage, VerificationOutcome


def _evaluation(subset: tuple[str, ...], *, passing: bool) -> SubsetEvaluation:
    return SubsetEvaluation(
        subset=subset,
        margins={"union-contract": 1.0 if passing else -1.0},
        outcome=VerificationOutcome.PASS if passing else VerificationOutcome.FAIL,
        violated_contracts=() if passing else ("union-contract",),
    )


def test_six_patch_exhaustive_audit_executes_exactly_63_nonempty_subsets() -> None:
    patches = tuple(f"p{index}" for index in range(6))
    calls: list[tuple[str, ...]] = []

    def oracle(subset: tuple[str, ...]) -> SubsetEvaluation:
        calls.append(subset)
        return _evaluation(subset, passing=True)

    result = audit_patch_pool(
        patches,
        oracle=oracle,
        config=AuditConfig(subset_budget=63, exhaustive_threshold=6, bootstrap_samples=2),
    )
    assert len(enumerate_subsets(patches)) == 63
    assert result.executed_subset_count == 63
    assert result.total_model_executions == 64  # separate empty-stack baseline
    assert calls[0] == ()
    assert result.claims == (AuditClaim.EXHAUSTIVE_COMPOSITION_AUDIT,)
    assert result.coverage is CompositionCoverage.EXHAUSTIVE_SUBSETS
    assert result.all_combinations_passed


def test_active_audit_finds_three_way_only_failure_and_ddmin_keeps_triple() -> None:
    patches = ("a", "b", "c", "d")
    trigger = {"a", "b", "c"}

    def oracle(subset: tuple[str, ...]) -> SubsetEvaluation:
        return _evaluation(subset, passing=not trigger <= set(subset))

    result = audit_patch_pool(
        patches,
        oracle=oracle,
        config=AuditConfig(
            subset_budget=11,
            maximum_order=3,
            exhaustive_threshold=0,
            include_all_pairs=True,
            active_batch_size=1,
            bootstrap_samples=3,
            cross_validation_folds=2,
            seed=4,
        ),
    )
    assert all(
        evaluation.passed for evaluation in result.evaluations if len(evaluation.subset) <= 2
    )
    assert ("a", "b", "c") in result.failing_subsets
    assert any(reduction.reduced == ("a", "b", "c") for reduction in result.minimal_failures)
    assert AuditClaim.FAILING_SUBSET_FOUND in result.claims
    assert AuditClaim.EXHAUSTIVE_COMPOSITION_AUDIT not in result.claims


def test_budgeted_no_failure_wording_never_claims_exhaustive() -> None:
    patches = ("a", "b", "c", "d")
    result = audit_patch_pool(
        patches,
        oracle=lambda subset: _evaluation(subset, passing=True),
        config=AuditConfig(
            subset_budget=4,
            exhaustive_threshold=0,
            include_all_pairs=False,
            bootstrap_samples=2,
        ),
    )
    assert result.claims == (
        AuditClaim.BUDGETED_COMPOSITION_AUDIT,
        AuditClaim.NO_FAILURE_FOUND_WITHIN_BUDGET,
    )
    assert not result.search_space_exhausted
    assert not result.all_combinations_passed


def test_ddmin_executes_candidates_and_returns_one_minimal_failure() -> None:
    calls: list[tuple[str, ...]] = []

    def fails(subset: tuple[str, ...]) -> bool:
        calls.append(subset)
        return {"a", "c"} <= set(subset)

    result = ddmin_failing_subset(("a", "b", "c", "d"), oracle=fails)
    assert result.reduced == ("a", "c")
    assert result.one_minimal
    assert calls
    assert all(candidate != ("a", "b", "c", "d") for candidate in calls)


def test_degree_three_surrogate_has_hierarchical_terms_and_fits_interaction() -> None:
    vectors = [tuple(vector) for vector in product((0, 1), repeat=3)]
    margins = [
        1.0 + 0.2 * vector[0] - 3.0 * vector[0] * vector[1] * vector[2] for vector in vectors
    ]
    terms = hierarchical_feature_terms(3, degree=3)
    assert (0,) in terms and (0, 1) in terms and (0, 1, 2) in terms
    model = PseudoBooleanRegressor(3, degree=3, alpha=0.0).fit(vectors, margins)
    assert model.predict_one((1, 1, 1)) == pytest.approx(-1.8, abs=1e-5)
    assert model.predict_one((1, 1, 0)) == pytest.approx(1.2, abs=1e-5)


def test_bootstrap_uncertainty_is_deterministic_for_fixed_seed() -> None:
    vectors = [(0, 0), (1, 0), (0, 1), (1, 1)]
    margins = [1.0, 0.8, 0.7, -0.2]
    first = fit_bootstrap_ensemble(
        vectors,
        margins,
        patch_count=2,
        degree=2,
        alpha=0.01,
        l1_ratio=0.8,
        samples=8,
        seed=99,
    ).predict_one((1, 1))
    second = fit_bootstrap_ensemble(
        vectors,
        margins,
        patch_count=2,
        degree=2,
        alpha=0.01,
        l1_ratio=0.8,
        samples=8,
        seed=99,
    ).predict_one((1, 1))
    assert first == second
