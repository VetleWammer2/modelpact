from __future__ import annotations

from pathlib import Path

from modelpact.modelpactbench.runner import run_selected
from modelpact.patch.ast import Alias
from modelpact.patch.bundle import MANDATORY_BUNDLE_ARTIFACTS, load_patch_bundle
from modelpact.verify.certificate import read_certificate, validate_certificate


def test_forkbench_executes_selective_extraction_and_emits_evidence(tmp_path: Path) -> None:
    output = tmp_path / "forkbench"
    result = run_selected("forkbench", artifact_output=output)

    assert result["suite"] == "ModelPactBench"
    assert result["status"] == "PASS"
    assert result["success"] is True
    diff = result["diff"]
    assert isinstance(diff, dict)
    assert diff["witness_count"] >= 5
    assert diff["cluster_count"] >= 5
    assert diff["minimized_witness_count"] >= 4
    assert diff["selected_witness_count"] >= 1
    assert diff["nonselected_witness_count"] >= 4
    assert result["model"]["learned_target_changes"] == 5

    cegis = result["cegis"]
    assert isinstance(cegis, dict)
    assert cegis["stop_reason"] == "NO_COUNTEREXAMPLE_WITHIN_BUDGET"
    rounds = cegis["rounds"]
    assert isinstance(rounds, list)
    assert any(item["target_counterexamples"] for item in rounds)
    assert any(item["guard_counterexamples"] for item in rounds)

    minimization = result["minimization"]
    assert isinstance(minimization, dict)
    assert minimization["verification_budget_used"] > 0
    assert minimization["candidates"]

    verification = result["verification"]
    assert isinstance(verification, dict)
    assert verification["outcome"] == "PASS"
    assert verification["holdout_outcome"] == "PASS"
    assert verification["holdout_accesses"] == 2
    assert verification["holdout_target_accesses"] == 1
    assert verification["holdout_guard_accesses"] == 1
    assert verification["holdout_target_assertions"] == 2
    assert verification["holdout_guard_assertions"] == 2
    assert verification["sealed_guard_probe_count"] == 4
    assert verification["selected_transfer_rate"] == 1.0
    assert verification["unselected_change_rejection_rate"] == 1.0
    assert verification["unchanged_control_preservation_rate"] == 1.0
    assert verification["free_generation_records"] > 0
    assert verification["worst_prompt_base_kl"] > 0.0

    patch_root = output / "patch"
    bundle = load_patch_bundle(patch_root)
    assert set(bundle.manifest.artifact_hashes) >= MANDATORY_BUNDLE_ARTIFACTS
    assert bundle.manifest.patch_id == result["patch"]["patch_id"]
    assert isinstance(bundle.program.targets["token_embedding.weight"], Alias)
    assert (patch_root / "apply_patch.py").is_file()
    assert (patch_root / "verify_patch.py").is_file()
    assert "import modelpact" not in (patch_root / "apply_patch.py").read_text(encoding="utf-8")
    assert "import modelpact" not in (patch_root / "verify_patch.py").read_text(encoding="utf-8")

    certificate = read_certificate(patch_root / "certificate.json")
    validate_certificate(certificate, artifact_root=patch_root)
    assert certificate.patch_id == bundle.manifest.patch_id
    assert "SEALED_HOLDOUT_VERIFIED" in certificate.claims
    assert "FREE_GENERATION_VERIFIED" in certificate.claims
    assert result["negative_findings"]
    assert (output / "difference" / "witnesses.parquet").is_file()
    assert (output / "result.json").is_file()
