"""Machine-readable ModelPactBench runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from modelpact.modelpactbench import (
    run_benign_collusion,
    run_closure_matrix,
    run_forkbench,
    run_huggingface_local,
    run_locality_cegis,
    run_r1_loop,
    run_semantic_merge,
    run_semantic_rebase,
)
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps

_FAILURE_STATUSES = frozenset({"FAIL", "ERROR", "INCONCLUSIVE", "UNSUPPORTED"})


def benchmark_succeeded(result: dict[str, object]) -> bool:
    """Interpret only explicit benchmark terminal fields, never benchmark metrics."""

    success = result.get("success")
    if not isinstance(success, bool):
        raise ValueError("benchmark success field must be boolean")
    status = result.get("status")
    if not isinstance(status, str):
        raise ValueError("benchmark status field must be a string")
    return success and status == "PASS" and status not in _FAILURE_STATUSES


def run_selected(
    name: str,
    *,
    artifact_output: str | Path | None = None,
) -> dict[str, object]:
    artifact_experiments = {
        "closure_matrix",
        "collusion",
        "forkbench",
        "huggingface_local",
        "r1_loop",
    }
    if artifact_output is not None and name not in artifact_experiments:
        raise ValueError("artifact output is supported only by retained model-execution benchmarks")
    functions: dict[str, Callable[[], dict[str, object]]] = {
        "closure_matrix": lambda: run_closure_matrix(artifact_output),
        "collusion": lambda: run_benign_collusion(artifact_output),
        "merge": run_semantic_merge,
        "rebase": run_semantic_rebase,
        "rebase_cross_architecture": lambda: run_semantic_rebase(cross_architecture=True),
        "cegis": run_locality_cegis,
        "forkbench": lambda: run_forkbench(artifact_output),
        "huggingface_local": lambda: run_huggingface_local(artifact_output),
        "r1_loop": lambda: run_r1_loop(artifact_output),
    }
    try:
        return functions[name]()
    except KeyError as error:
        raise ValueError(f"unknown ModelPactBench experiment: {name}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic ModelPactBench experiment")
    parser.add_argument(
        "name",
        choices=(
            "closure_matrix",
            "collusion",
            "merge",
            "rebase",
            "rebase_cross_architecture",
            "cegis",
            "forkbench",
            "huggingface_local",
            "r1_loop",
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="retain the complete artifact tree for model-execution benchmarks",
    )
    arguments = parser.parse_args()
    result = run_selected(arguments.name, artifact_output=arguments.artifacts)
    rendered = canonical_dumps(result) + "\n"
    if arguments.output is not None:
        atomic_write_text(arguments.output, rendered)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not benchmark_succeeded(result):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
