from __future__ import annotations

import json
from pathlib import Path

from modelpact.modelpactbench.r1_loop import run_r1_loop
from modelpact.util.canonical_json import canonical_dumps


def _section(result: dict[str, object], name: str) -> dict[str, object]:
    value = result[name]
    assert isinstance(value, dict)
    return value


def test_unified_r1_loop_executes_conflict_merge_rebase_and_revert(tmp_path: Path) -> None:
    output = tmp_path / "r1-loop"
    result = run_r1_loop(output)

    assert result["suite"] == "ModelPactBench"
    assert result["benchmark"] == "R1 Unified Tiny Loop"
    assert result["status"] == "PASS"
    assert result["success"] is True
    assert "wall_seconds" not in canonical_dumps(result)

    persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert persisted == result
    assert canonical_dumps(persisted) == canonical_dumps(result)
    assert (output / "checkpoints" / "base" / "model.safetensors").is_file()
    assert (output / "checkpoints" / "target" / "model.safetensors").is_file()
    assert (output / "checkpoints" / "base-v2" / "model.safetensors").is_file()

    behavioral_diff = _section(result, "behavioral_diff")
    assert behavioral_diff["prompts_evaluated"] == 6
    assert behavioral_diff["witness_count"] >= 5
    assert behavioral_diff["cluster_count"] >= 4

    extraction = _section(result, "first_patch_extraction")
    assert extraction["validation_passed"] is True
    assert extraction["status"] == "FEASIBLE"
    assert extraction["optimization_steps_executed"] > 0
    assert extraction["primal_dual"] is True
    assert extraction["contract_outcome"] == "PASS"
    assert extraction["outputs"]["F:a"] == "G"
    assert extraction["outputs"]["P"] == "X"
    assert extraction["outputs"]["L"] == "M"

    second = _section(result, "second_patch_compilation")
    assert second["status"] == "FEASIBLE"
    assert second["optimization_steps_executed"] > 0
    assert second["primal_dual"] is True
    assert second["contract_outcome"] == "PASS"
    assert second["outputs"]["P"] == "{"
    assert second["outputs"]["F:a"] == "R"

    runtime = _section(result, "runtime_mount")
    assert runtime["mounted_outputs"]["F:a"] == "G"
    assert runtime["unmounted_outputs"]["F:a"] == "R"
    assert runtime["unmount_exact"] is True
    assert runtime["base_source_unchanged"] is True

    composition = _section(result, "composition")
    assert composition["individual_contracts_pass"] is True
    assert composition["naive_claim"] == "SEMANTIC_CONFLICT"
    assert composition["failed_contracts"]
    assert composition["naive_outputs"]["F:a"] != "G" or composition["naive_outputs"]["P"] != "{"

    merge = _section(result, "semantic_merge")
    assert merge["disposition"] == "SEMANTIC_MERGE_VERIFIED"
    assert merge["claim"] == "COMPOSITION_CLOSED"
    assert merge["compiler_invoked"] is True
    assert merge["optimization_steps_executed"] > 0
    assert merge["fresh_delta_differs_from_parent_sum"] is True
    assert merge["verification_outcome"] == "PASS"
    assert merge["outputs"]["F:a"] == "G"
    assert merge["outputs"]["P"] == "{"
    assert merge["outputs"]["L"] == "M"

    rebase = _section(result, "semantic_rebase")
    assert rebase["direct_attempted"] is True
    assert rebase["direct_verified"] is False
    assert rebase["direct_outcome"] == "FAIL"
    assert rebase["claim"] == "SEMANTIC_REBASE_VERIFIED"
    assert rebase["disposition"] == "SEMANTIC_REBASE_VERIFIED"
    assert rebase["recompile_optimization_steps"] > 0
    assert rebase["base_v2_unpatched_outputs"]["F:b"] == "D"
    assert rebase["final_outputs"]["F:a"] == "G"
    assert rebase["final_outputs"]["F:b"] == "D"
    assert rebase["final_outputs"]["P"] == "X"

    stack = _section(result, "stack_and_revert")
    assert stack["resolved_kind"] == "VERIFIED_COMPOSITE_PATCH"
    assert stack["remaining_resolution"] == "NAIVE_ADDITIVE_STACK"
    assert stack["remaining_contract_outcome"] == "PASS"
    assert stack["reversion_grade"] == "VERIFIED_LOGICAL_STACK_RECONSTRUCTED"
    assert stack["numeric_subtraction_used"] is False
    assert stack["remaining_outputs"]["F:a"] == "G"
    assert stack["remaining_outputs"]["P"] == "X"

    integrity = _section(result, "source_integrity")
    assert integrity == {
        "base_unchanged": True,
        "target_unchanged": True,
        "base_v2_unchanged_after_rebase": True,
    }
    evidence = _section(result, "evidence_scope")
    assert evidence["contract_coverage"] == "FINITE_PARTIAL"
    assert "SEALED_HOLDOUT_VERIFIED" in evidence["unsupported_claims"]
