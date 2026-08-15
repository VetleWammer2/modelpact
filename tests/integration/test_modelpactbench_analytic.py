from __future__ import annotations

from pathlib import Path

from modelpact.modelpactbench import (
    run_benign_collusion,
    run_closure_matrix,
    run_locality_cegis,
    run_semantic_merge,
    run_semantic_rebase,
)
from modelpact.patch.bundle import load_patch_bundle


def test_closure_matrix_executes_all_63_tiny_causal_lm_subsets(tmp_path: Path) -> None:
    output = tmp_path / "closure-matrix"
    result = run_closure_matrix(output)
    assert result["executed_subsets"] == 63
    assert result["search_space_exhausted"] is True
    assert result["individual_patches_pass"] is True
    assert result["model"] == "TinyCausalLM"
    assert result["ground_truth_failure_count"] == 8
    assert result["minimal_interaction_order"] == 3
    ground_truth = result["subset_ground_truth"]
    assert isinstance(ground_truth, list)
    assert len(ground_truth) == 63
    assert all(row["result_hash"].startswith("sha256:") for row in ground_truth)
    triple = next(
        row for row in ground_truth if row["subset"] == ["behavior-0", "behavior-1", "behavior-2"]
    )
    assert triple["outcome"] == "FAIL"
    assert (
        triple["metadata"]["generated_token_ids"]["synthetic combined format:"]
        == triple["metadata"]["trigger_failure_token"]
    )
    patches = result["patches"]
    assert isinstance(patches, list)
    assert len(patches) == 6
    assert [item["rank"] for item in patches] == [2, 2, 2, 1, 1, 1]
    assert len({item["content_id"] for item in patches}) == 6
    for patch in patches:
        bundle = load_patch_bundle(output / "patches" / patch["patch_id"])
        assert bundle.manifest.patch_id == patch["content_id"]
        assert bundle.manifest.compiler_configuration["guard_prompt_count"] == 2
    assert (output / "base-checkpoint" / "model.safetensors").is_file()
    assert (output / "result.json").is_file()

    repeated = run_closure_matrix()
    assert repeated["patches"] == result["patches"]
    assert repeated["subset_ground_truth"] == result["subset_ground_truth"]
    assert repeated["failing_subsets"] == result["failing_subsets"]


def test_benign_collusion_is_three_way_and_found(tmp_path: Path) -> None:
    result = run_benign_collusion(tmp_path / "collusion")
    assert result["individual_patches_pass"] is True
    assert result["relevant_pairs_pass"] is True
    assert result["three_way_ground_truth_fails"] is True
    assert result["failing_subsets"]
    assert ["behavior-0", "behavior-1", "behavior-2"] in result["minimal_failures"]
    assert result["coverage"] == "ACTIVE_BUDGETED"
    assert result["executed_subsets"] < result["possible_nonempty_subsets"]
    assert result["ground_truth_failure_count"] == 8
    baselines = result["baseline_comparison"]
    assert baselines["singleton_only"]["failure_found"] is False
    assert baselines["pairwise_only"]["failure_found"] is False
    assert baselines["pairwise_only"]["false_assurance_rate"] == 1.0
    assert baselines["active_sparse_interaction"]["failure_found"] is True
    assert baselines["active_sparse_interaction"]["minimal_failing_subset_recovered"] is True
    assert result["active_proposals"]
    assert result["negative_findings"]


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
