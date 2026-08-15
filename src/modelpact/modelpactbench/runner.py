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
    run_locality_cegis,
    run_semantic_merge,
    run_semantic_rebase,
)
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps


def run_selected(
    name: str,
    *,
    artifact_output: str | Path | None = None,
) -> dict[str, object]:
    if artifact_output is not None and name != "forkbench":
        raise ValueError("artifact output is currently supported only by ForkBench")
    functions: dict[str, Callable[[], dict[str, object]]] = {
        "closure_matrix": run_closure_matrix,
        "collusion": run_benign_collusion,
        "merge": run_semantic_merge,
        "rebase": run_semantic_rebase,
        "rebase_cross_architecture": lambda: run_semantic_rebase(cross_architecture=True),
        "cegis": run_locality_cegis,
        "forkbench": lambda: run_forkbench(artifact_output),
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
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="retain the complete artifact tree (currently used by ForkBench)",
    )
    arguments = parser.parse_args()
    result = run_selected(arguments.name, artifact_output=arguments.artifacts)
    rendered = canonical_dumps(result) + "\n"
    if arguments.output is not None:
        atomic_write_text(arguments.output, rendered)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
