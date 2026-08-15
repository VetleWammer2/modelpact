"""Print a concise transcript derived only from committed benchmark JSON."""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from modelpact.patch.bundle import load_patch_bundle, require_complete_bundle
from modelpact.util.canonical_json import strict_json_loads
from modelpact.util.hashing import hash_canonical
from modelpact.verify.certificate import read_certificate, validate_certificate

MAX_SUMMARY_ARTIFACT_BYTES = 64 * 1024 * 1024


def _load(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"benchmark artifact is not a regular file: {name}")
    if path.stat().st_size > MAX_SUMMARY_ARTIFACT_BYTES:
        raise ValueError(f"benchmark artifact exceeds size limit: {name}")
    value = strict_json_loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"benchmark artifact is not an object: {name}")
    return value


def _require_success(value: dict[str, Any], *, label: str) -> None:
    if value.get("status") != "PASS" or value.get("success") is not True:
        raise ValueError(f"benchmark artifact is not a successful terminal result: {label}")


def _result(root: Path, name: str, experiment: str) -> dict[str, Any]:
    wrapper = _load(root, name)
    _require_success(wrapper, label=name)
    results = wrapper.get("results")
    if not isinstance(results, dict) or not isinstance(results.get(experiment), dict):
        raise ValueError(f"benchmark artifact omits experiment {experiment}: {name}")
    result = results[experiment]
    _require_success(result, label=f"{name}:{experiment}")
    return result


def _validate_forkbench_artifacts(root: Path, fork: dict[str, Any]) -> None:
    retained = _load(root / "forkbench-run", "result.json")
    if retained != fork:
        raise ValueError("forkbench.json differs from forkbench-run/result.json")
    patch_root = root / "forkbench-run" / "patch"
    bundle = load_patch_bundle(patch_root)
    require_complete_bundle(bundle.manifest)
    patch = fork.get("patch")
    if not isinstance(patch, dict) or patch.get("patch_id") != bundle.manifest.patch_id:
        raise ValueError("ForkBench result patch identity does not match retained bundle")
    certificate = read_certificate(patch_root / "certificate.json")
    validate_certificate(certificate, artifact_root=patch_root)
    if certificate.patch_id != bundle.manifest.patch_id:
        raise ValueError("ForkBench certificate patch identity mismatch")
    standalone = _load(root, "forkbench_standalone_verify.json")
    result_hash = standalone.get("result_hash")
    unhashed = {key: value for key, value in standalone.items() if key != "result_hash"}
    if result_hash != hash_canonical(unhashed):
        raise ValueError("ForkBench standalone report hash mismatch")
    if standalone.get("outcome") != "PASS":
        raise ValueError("ForkBench standalone verification is not PASS")
    if standalone.get("patch_id") != bundle.manifest.patch_id:
        raise ValueError("ForkBench standalone patch identity mismatch")
    if standalone.get("artifact_hashes") != dict(bundle.manifest.artifact_hashes):
        raise ValueError("ForkBench standalone artifact hashes do not match retained bundle")
    environment = standalone.get("environment")
    if not isinstance(environment, dict) or environment.get("modelpact_importable") is not False:
        raise ValueError("ForkBench standalone report lacks package-isolation evidence")
    if (
        environment.get("python_no_site") is not True
        or environment.get("python_safe_path") is not True
    ):
        raise ValueError("ForkBench standalone report lacks -S/-P interpreter evidence")


def _render(root: Path) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        _print_summary(root)
    return stream.getvalue()


def _print_summary(root: Path) -> None:
    """Print the validated summary to the current standard output."""

    fork = _load(root, "forkbench.json")
    closure = _result(root, "closure_matrix.json", "closure_matrix")
    collusion = _result(root, "collusion.json", "collusion")
    cegis = _result(root, "cegis.json", "cegis")
    loop = _load(root, "r1_loop.json")
    huggingface = _load(root, "huggingface_local.json")
    environment = _load(root, "environment.json")
    materialization = _load(root / "materialization-run", "materialization-manifest.json")

    for label, result in (
        ("forkbench.json", fork),
        ("r1_loop.json", loop),
        ("huggingface_local.json", huggingface),
    ):
        _require_success(result, label=label)
    _validate_forkbench_artifacts(root, fork)

    fork_diff = fork["diff"]
    fork_patch = fork["patch"]
    fork_verify = fork["verification"]
    print("ModelPact R1 executed evidence")
    print(
        "ForkBench"
        f" status={fork['status']} witnesses={fork_diff['witness_count']}"
        f" clusters={fork_diff['cluster_count']} rank={fork['extraction']['total_rank']}"
        f" factor_bytes={fork_patch['factor_tensor_bytes']}"
        f" target={fork_verify['selected_transfer_rate']:.3f}"
        f" unselected_rejection={fork_verify['unselected_change_rejection_rate']:.3f}"
        f" holdout={fork_verify['holdout_outcome']}"
    )
    print(
        "R1Loop"
        f" status={loop['status']} naive={loop['composition']['naive_claim']}"
        f" merge={loop['semantic_merge']['disposition']}"
        f" rebase={loop['semantic_rebase']['disposition']}"
        f" revert={loop['stack_and_revert']['reversion_grade']}"
    )
    print(
        "ClosureMatrix"
        f" status={closure['status']} model={closure['model']}"
        f" subsets={closure['executed_subsets']}/{closure['possible_nonempty_subsets']}"
        f" failures={len(closure['failing_subsets'])} coverage={closure['coverage']}"
    )
    active = collusion["baseline_comparison"]["active_sparse_interaction"]
    pairwise = collusion["baseline_comparison"]["pairwise_only"]
    print(
        "BenignCollusion"
        f" status={collusion['status']} model={collusion['model']}"
        f" subsets={collusion['executed_subsets']}/{collusion['possible_nonempty_subsets']}"
        f" active_found={str(active['failure_found']).lower()}"
        f" pairwise_found={str(pairwise['failure_found']).lower()}"
        f" minimal_order={min(map(len, collusion['minimal_failures']))}"
    )
    standalone = huggingface["standalone_verification"]
    standalone_passes = sum(item["outcome"] == "PASS" for item in standalone.values())
    print(
        "HuggingFaceLocal"
        f" status={huggingface['status']} patches={len(huggingface['patches'])}"
        f" compose={huggingface['composition']['claim']}"
        f" rebase={huggingface['rebase']['claim']}"
        f" standalone={standalone_passes}/{len(standalone)}"
    )
    materialization_performance = materialization["performance"]
    print(
        "Materialization"
        f" strategy={materialization_performance['streaming_strategy']}"
        f" shards={len(materialization['output_files']) - 1}"
        f" read_bytes={materialization_performance['read_bytes']}"
        f" write_bytes={materialization_performance['write_bytes']}"
        f" peak_rss={materialization_performance['peak_rss_bytes']}"
    )
    print(
        "CEGIS"
        f" status={cegis['status']} negative_result={str(cegis['negative_result']).lower()}"
        f" search_failures={cegis['initial_search_failures']}"
        f"->{cegis['post_cegis_search_failures']}"
    )
    print(
        "Environment"
        f" python={environment['python']['version']} torch={environment['pytorch']['version']}"
        f" gpu_executed={str(environment['gpu_execution_performed']).lower()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("research/artifacts"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.root
    rendered = _render(root)
    if arguments.output is not None:
        from modelpact.util.atomic import atomic_write_text

        atomic_write_text(arguments.output, rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
