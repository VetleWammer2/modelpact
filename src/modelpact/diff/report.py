"""Deterministic difference bundle writer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from modelpact.diff.cluster import WitnessCluster
from modelpact.diff.witnesses import DifferenceWitness, witness_set_hash
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical


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
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet difference bundles require the 'parquet' extra") from error
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "witnesses.parquet", compression="zstd")
    atomic_write_text(root / "clusters.json", canonical_dumps([asdict(item) for item in clusters]) + "\n")
    atomic_write_text(
        root / "minimized-prompts.jsonl",
        "".join(json.dumps({"witness_id": item.witness_id, "prompt": item.minimized_input}, ensure_ascii=False, sort_keys=True) + "\n" for item in ordered),
    )
    report_lines = [
        "# Scoped behavioral difference report",
        "",
        f"Observed witnesses: {len(ordered)}",
        f"Empirical clusters: {len(clusters)}",
        "",
        "These witnesses describe only the executed probe space and search budget. They do not establish complete model equivalence or difference.",
    ]
    atomic_write_text(root / "report.md", "\n".join(report_lines) + "\n")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "scope": "executed_probe_space",
        "witness_set_hash": witness_set_hash(ordered),
        "cluster_hash": hash_canonical([asdict(item) for item in clusters]),
        "configuration": configuration,
        "warnings": ["Scoped witnesses are not a complete description of model differences."],
    }
    atomic_write_text(root / "manifest.json", canonical_dumps(manifest) + "\n")
    return manifest

