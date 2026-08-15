from __future__ import annotations

from modelpact.modelpactbench import (
    run_benign_collusion,
    run_closure_matrix,
    run_locality_cegis,
    run_semantic_merge,
    run_semantic_rebase,
)


def test_closure_matrix_executes_all_63_subsets() -> None:
    result = run_closure_matrix()
    assert result["executed_subsets"] == 63
    assert result["search_space_exhausted"] is True
    assert result["individual_patches_pass"] is True


def test_benign_collusion_is_three_way_and_found() -> None:
    result = run_benign_collusion()
    assert result["individual_patches_pass"] is True
    assert result["relevant_pairs_pass"] is True
    assert result["three_way_ground_truth_fails"] is True
    assert result["failing_subsets"]
    assert ["field-a", "field-b", "field-c"] in result["minimal_failures"]
    assert result["coverage"] == "ACTIVE_BUDGETED"


def test_semantic_merge_runs_new_optimization() -> None:
    result = run_semantic_merge()
    assert result["parents_individually_pass"] is True
    assert result["naive_passed"] is False
    assert result["compiler_invoked"] is True
    assert result["merged_verified"] is True
    assert result["optimization_steps"] > 0
    baselines = result["baseline_comparison"]
    assert isinstance(baselines, dict)
    assert set(baselines) == {
        "cat_style_projection",
        "dare",
        "joint_multitask_low_rank",
        "naive_delta_sum",
        "task_arithmetic",
        "ties",
        "weighted_delta_sum",
    }
    assert baselines["naive_delta_sum"]["passed"] is False
    assert baselines["weighted_delta_sum"]["passed"] is True
    assert result["negative_findings"]


def test_semantic_rebase_recompiles_failed_direct_transfer() -> None:
    result = run_semantic_rebase()
    assert result["direct_attempted"] is True
    assert result["direct_verified"] is False
    assert result["recompiled"] is True
    assert result["verified"] is True


def test_cross_architecture_rebase_never_transplants_tensors() -> None:
    result = run_semantic_rebase(cross_architecture=True)
    assert result["direct_attempted"] is False
    assert result["recompiled"] is True
    assert result["verified"] is True


def test_cegis_reports_negative_case_without_claiming_improvement() -> None:
    result = run_locality_cegis()
    assert result["initial_search_failures"] > 0
    assert result["counterexamples_added"]
    assert result["post_cegis_search_failures"] > 0
    assert result["negative_result"] is True
