"""Execute one committed ModelPactBench configuration and write raw JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelpact.modelpactbench.runner import benchmark_succeeded, run_selected
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical

SUPPORTED_EXPERIMENTS = frozenset(
    {
        "cegis",
        "closure_matrix",
        "collusion",
        "forkbench",
        "merge",
        "r1_loop",
        "rebase",
        "rebase_cross_architecture",
    }
)


def _load_configuration(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("benchmark configuration must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark configuration must be a JSON object")
    allowed = {"schema_version", "experiments", "offline", "device"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown benchmark configuration fields: {sorted(unknown)}")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported benchmark configuration version")
    if value.get("offline") is not True or value.get("device") != "cpu":
        raise ValueError("the committed analytic runner requires offline CPU execution")
    experiments = value.get("experiments")
    if (
        not isinstance(experiments, list)
        or not experiments
        or not all(isinstance(item, str) for item in experiments)
        or experiments != sorted(set(experiments))
    ):
        raise ValueError("experiments must be a nonempty sorted unique string list")
    unsupported = set(experiments) - SUPPORTED_EXPERIMENTS
    if unsupported:
        raise ValueError(f"unsupported ModelPactBench experiments: {sorted(unsupported)}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = _load_configuration(arguments.config)
    experiments = configuration["experiments"]
    assert isinstance(experiments, list)
    results = {name: run_selected(name) for name in experiments}
    success = all(benchmark_succeeded(result) for result in results.values())
    payload = {
        "schema_version": 1,
        "configuration": configuration,
        "configuration_hash": hash_canonical(configuration),
        "results": results,
        "status": "PASS" if success else "FAIL",
        "success": success,
    }
    rendered = canonical_dumps(payload) + "\n"
    atomic_write_text(arguments.output, rendered)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
