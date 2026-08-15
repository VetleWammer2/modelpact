from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import packaging
import safetensors
import torch
import yaml

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
    assert not (patch_root / "tiny_adapter.py").exists()
    assert "import modelpact" not in (patch_root / "apply_patch.py").read_text(encoding="utf-8")
    assert "import modelpact" not in (patch_root / "verify_patch.py").read_text(encoding="utf-8")

    dependency_roots = {
        Path(module.__file__).resolve().parents[1]
        for module in (numpy, packaging, safetensors, torch, yaml)
    }
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in sorted(dependency_roots))
    environment["PYTHONNOUSERSITE"] = "1"
    preflight = subprocess.run(  # noqa: S603 - exact interpreter and constant code
        [
            sys.executable,
            "-S",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('modelpact') is None",
        ],
        cwd=patch_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert preflight.returncode == 0, preflight.stderr
    independent = subprocess.run(  # noqa: S603 - executes exact generated script path
        [
            sys.executable,
            "-S",
            "-P",
            str(patch_root / "verify_patch.py"),
            str(output / "base-checkpoint"),
            "--adapter-kind",
            "tiny",
            "--include-holdout",
        ],
        cwd=patch_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert independent.returncode == 0, independent.stdout + independent.stderr
    independent_result = json.loads(independent.stdout)
    assert independent_result["outcome"] == "PASS"
    assert independent_result["environment"]["modelpact_importable"] is False
    assert independent_result["environment"]["python_no_site"] is True
    assert independent_result["environment"]["python_safe_path"] is True
    assert independent_result["model_adapter_id"] == "modelpact.tiny_causal_lm.v1"
    assert independent_result["unsupported_claims"] == []
    independent_assertions = independent_result["verification_results"]
    assert all(item["outcome"] == "PASS" for item in independent_assertions)
    assert {item["assertion_type"] for item in independent_assertions} >= {
        "base_kl",
        "free_generation_match",
        "reference_kl",
    }

    certificate = read_certificate(patch_root / "certificate.json")
    validate_certificate(certificate, artifact_root=patch_root)
    assert certificate.patch_id == bundle.manifest.patch_id
    assert "SEALED_HOLDOUT_VERIFIED" in certificate.claims
    assert "FREE_GENERATION_VERIFIED" in certificate.claims
    assert result["negative_findings"]
    assert (output / "difference" / "witnesses.parquet").is_file()
    assert (output / "result.json").is_file()
