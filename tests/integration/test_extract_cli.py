from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import torch
from typer.testing import CliRunner

import modelpact.cli as cli_module
from modelpact.cli import (
    _load_extraction_clusters,
    _load_extraction_witnesses,
    _witness_from_row,
    app,
)
from modelpact.diff.witnesses import DifferenceWitness
from modelpact.models.manifest import build_model_manifest
from modelpact.patch.bundle import load_patch_bundle
from modelpact.util.hashing import sha256_file
from tests.support.extraction_adapter import (
    ExtractionTestAdapter,
    save_extraction_checkpoint,
)

RUNNER = CliRunner()


def _write_diff_bundle(path: Path, *, base: Path, target: Path) -> None:
    adapter = ExtractionTestAdapter()
    base_model = adapter.load(str(base), device="cpu", dtype=torch.float32)
    target_model = adapter.load(str(target), device="cpu", dtype=torch.float32)
    base_manifest = build_model_manifest(
        base_model,
        checkpoint=base,
        adapter_id=adapter.adapter_id,
    )
    target_manifest = build_model_manifest(
        target_model,
        checkpoint=target,
        adapter_id=adapter.adapter_id,
    )
    selected = DifferenceWitness.create(
        original_input="Please answer TARGET.",
        minimized_input="TARGET",
        divergence_metrics={"next_token_kl": 1.0},
        base_output="A",
        target_output="B",
        provenance={"domain": "selected"},
    )
    guard = DifferenceWitness.create(
        original_input="Please answer GUARD.",
        minimized_input="GUARD",
        divergence_metrics={"next_token_kl": 1.0},
        base_output="A",
        target_output="B",
        provenance={"domain": "nonselected"},
    )
    path.mkdir()
    clusters = [
        {
            "cluster_id": "cluster-target",
            "dispersion": 0.0,
            "medoid_id": selected.witness_id,
            "outlier_ids": [],
            "uncertainty": 0.0,
            "witness_ids": [selected.witness_id],
        },
        {
            "cluster_id": "cluster-unselected",
            "dispersion": 0.0,
            "medoid_id": guard.witness_id,
            "outlier_ids": [],
            "uncertainty": 0.0,
            "witness_ids": [guard.witness_id],
        },
    ]
    (path / "clusters.json").write_text(json.dumps(clusters, sort_keys=True), encoding="utf-8")
    parquet.write_table(
        pa.Table.from_pylist([selected.to_dict(), guard.to_dict()]),
        path / "witnesses.parquet",
    )
    manifest = {
        "artifact_hashes": {
            "clusters.json": sha256_file(path / "clusters.json"),
            "witnesses.parquet": sha256_file(path / "witnesses.parquet"),
        },
        "configuration": {
            "base_signature": base_manifest.signature.to_dict(),
            "target_signature": target_manifest.signature.to_dict(),
        },
        "schema_version": 1,
        "witness_set_hash": "sha256:" + "1" * 64,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_extract_cli_executes_disjoint_cegis_minimization_and_one_holdout(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    target = tmp_path / "target"
    diff = tmp_path / "diff"
    output = tmp_path / "patch"
    save_extraction_checkpoint(base, target=False)
    save_extraction_checkpoint(target, target=True)
    _write_diff_bundle(diff, base=base, target=target)

    result = RUNNER.invoke(
        app,
        [
            "extract",
            str(diff),
            "--cluster",
            "cluster-target",
            "--base",
            "tests.support.extraction_adapter:ExtractionTestAdapter",
            "--base-checkpoint",
            str(base),
            "--target",
            "tests.support.extraction_adapter:ExtractionTestAdapter",
            "--target-checkpoint",
            str(target),
            "--output",
            str(output),
            "--max-rank",
            "1",
            "--max-modules",
            "1",
            "--steps",
            "240",
            "--cegis-rounds",
            "1",
            "--search-budget",
            "1",
            "--validation-probes",
            "1",
            "--holdout-probes",
            "1",
            "--minimization-budget",
            "2",
            "--max-new-tokens",
            "1",
            "--seed",
            "11",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["extraction"]["holdout_candidate_executions"] == 1
    assert payload["verification"]["holdout_outcome"] == "PASS"
    role_evidence = payload["extraction"]["prompt_roles"]
    role_hashes = [
        digest
        for role, evidence in role_evidence.items()
        if role != "schema_version"
        for digest in evidence["prompt_hashes"]
    ]
    assert len(role_hashes) == len(set(role_hashes))
    minimization = json.loads(
        (output / "evidence" / "minimization.json").read_text(encoding="utf-8")
    )
    assert minimization["verification_budget_used"] == 2
    holdout = json.loads((output / "evidence" / "holdout.json").read_text(encoding="utf-8"))
    assert holdout["candidate_evaluations"] == 1
    assert holdout["outcome"] == "PASS"
    assert (output / "apply_patch.py").is_file()
    assert (output / "verify_patch.py").is_file()
    bundle = load_patch_bundle(output)
    assert bundle.evidence_id in (output / "apply_patch.py").read_text(encoding="utf-8")
    assert bundle.evidence_id in (output / "verify_patch.py").read_text(encoding="utf-8")


def test_extraction_rejects_oversized_cluster_and_parquet_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clusters = tmp_path / "clusters.json"
    clusters.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_DIFF_MAX_CLUSTERS_BYTES", 1)
    with pytest.raises(ValueError, match="oversized"):
        _load_extraction_clusters(clusters)

    witness = DifferenceWitness.create(
        original_input="TARGET",
        minimized_input="TARGET",
        divergence_metrics={"kl": 1.0},
        base_output="A",
        target_output="B",
        provenance={"domain": "selected"},
    )
    table = tmp_path / "witnesses.parquet"
    parquet.write_table(pa.Table.from_pylist([witness.to_dict()]), table)
    monkeypatch.setattr(cli_module, "_DIFF_MAX_WITNESSES_BYTES", 1)
    with pytest.raises(ValueError, match="oversized"):
        _load_extraction_witnesses(table)


def test_extraction_rejects_nonfinite_and_excessively_nested_witness_rows() -> None:
    witness = DifferenceWitness.create(
        original_input="TARGET",
        minimized_input="TARGET",
        divergence_metrics={"kl": 1.0},
        base_output="A",
        target_output="B",
        provenance={"domain": "selected"},
    ).to_dict()
    witness["gradient_fingerprint"] = [float("nan")]
    with pytest.raises(ValueError, match="non-finite"):
        _witness_from_row(witness)

    nested: object = "leaf"
    for index in range(10):
        nested = {f"level-{index}": nested}
    witness["gradient_fingerprint"] = []
    witness["provenance"] = nested
    with pytest.raises(ValueError, match="nested value limits"):
        _witness_from_row(witness)
