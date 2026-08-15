"""Deterministic difference bundle writer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from modelpact.checkpoints.safetensors import save_safetensors_atomic
from modelpact.diff.cluster import WitnessCluster
from modelpact.diff.witnesses import DifferenceWitness, witness_set_hash
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_file


def write_difference_bundle(
    output: str | Path,
    witnesses: tuple[DifferenceWitness, ...],
    clusters: tuple[WitnessCluster, ...],
    *,
    configuration: dict[str, object],
) -> dict[str, object]:
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"difference output must be empty: {root}")
    (root / "cluster-reports").mkdir(parents=True, exist_ok=True)
    ordered = tuple(sorted(witnesses, key=lambda item: item.witness_id))
    rows = [item.to_dict() for item in ordered]
    try:
        import pyarrow as pa  # type: ignore[import-untyped]  # Upstream has no py.typed.
        import pyarrow.parquet as pq  # type: ignore[import-untyped]  # Upstream stub gap.
    except ImportError as error:
        raise RuntimeError("Parquet difference bundles require the 'parquet' extra") from error
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "witnesses.parquet", compression="zstd")
    cluster_rows = [asdict(item) for item in clusters]
    atomic_write_text(root / "clusters.json", canonical_dumps(cluster_rows) + "\n")
    atomic_write_text(
        root / "minimized-prompts.jsonl",
        "".join(
            json.dumps(
                {"prompt": item.minimized_input, "witness_id": item.witness_id},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for item in ordered
        ),
    )
    tensor_keys = {item.witness_id: f"witness_{index:08d}" for index, item in enumerate(ordered)}
    activation_tensors = {
        tensor_keys[item.witness_id]: torch.tensor(item.activation_fingerprint, dtype=torch.float64)
        for item in ordered
    }
    gradient_tensors = {
        tensor_keys[item.witness_id]: torch.tensor(item.gradient_fingerprint, dtype=torch.float64)
        for item in ordered
    }
    if not activation_tensors:
        activation_tensors = {"_empty": torch.empty(0, dtype=torch.float64)}
        gradient_tensors = {"_empty": torch.empty(0, dtype=torch.float64)}
    tensor_metadata = {"schema_version": "1", "witness_set_hash": witness_set_hash(ordered)}
    save_safetensors_atomic(
        root / "activation-fingerprints.safetensors",
        activation_tensors,
        metadata=tensor_metadata,
    )
    save_safetensors_atomic(
        root / "gradient-fingerprints.safetensors",
        gradient_tensors,
        metadata=tensor_metadata,
    )
    for cluster in sorted(clusters, key=lambda item: item.cluster_id):
        members = [item.to_dict() for item in ordered if item.witness_id in cluster.witness_ids]
        cluster_report = {
            "cluster": asdict(cluster),
            "scope": "executed_difference_witnesses",
            "witnesses": members,
        }
        atomic_write_text(
            root / "cluster-reports" / f"{cluster.cluster_id}.json",
            canonical_dumps(cluster_report) + "\n",
        )
    report_lines = [
        "# Scoped behavioral difference report",
        "",
        f"Observed witnesses: {len(ordered)}",
        f"Empirical clusters: {len(clusters)}",
        "",
        "These witnesses describe only the executed probe space and search budget. "
        "They do not establish complete model equivalence or difference.",
    ]
    atomic_write_text(root / "report.md", "\n".join(report_lines) + "\n")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "scope": "executed_probe_space",
        "witness_set_hash": witness_set_hash(ordered),
        "cluster_hash": hash_canonical(cluster_rows),
        "configuration": configuration,
        "fingerprint_tensor_keys": tensor_keys,
        "warnings": ["Scoped witnesses are not a complete description of model differences."],
    }
    artifact_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest["artifact_hashes"] = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in artifact_paths
    }
    atomic_write_text(root / "manifest.json", canonical_dumps(manifest) + "\n")
    return manifest
