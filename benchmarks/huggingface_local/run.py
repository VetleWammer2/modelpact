"""Offline local Hugging Face manifest and free-generation preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import torch

from modelpact.adapters.base import GenerationPolicy
from modelpact.adapters.huggingface import HuggingFaceCausalLMAdapter
from modelpact.models.manifest import build_model_manifest
from modelpact.probes.dataset import ProbeRole, load_jsonl, probes_hash
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_bytes, sha256_file
from modelpact.util.paths import resolve_inside

ADAPTER_ID = "modelpact.huggingface_causal_lm.local.v1"


def _configuration(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("configuration must be a regular JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    allowed = {
        "adapter",
        "device",
        "dtype",
        "generation",
        "offline",
        "probe_source",
        "schema_version",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    if value.get("schema_version") != 1 or value.get("offline") is not True:
        raise ValueError("the local Hugging Face runner requires offline config version 1")
    if value.get("adapter") != ADAPTER_ID:
        raise ValueError("the configuration does not select the built-in safe local adapter")
    if value.get("dtype") not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if not isinstance(value.get("device"), str) or not value["device"]:
        raise ValueError("device must be a nonempty string")
    if not isinstance(value.get("probe_source"), str):
        raise ValueError("probe_source must be a relative path")
    generation = value.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    generation_unknown = set(generation) - {"max_new_tokens", "mode", "seed", "temperature"}
    if generation_unknown:
        raise ValueError(f"unknown generation fields: {sorted(generation_unknown)}")
    if generation.get("mode") not in {"greedy", "sample"}:
        raise ValueError("generation mode must be greedy or sample")
    maximum = generation.get("max_new_tokens")
    seed = generation.get("seed")
    temperature = generation.get("temperature", 1.0)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 4096:
        raise ValueError("max_new_tokens is outside supported bounds")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("generation seed is outside supported bounds")
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise ValueError("temperature must be numeric")
    return value


def _dtype(name: object) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    assert isinstance(name, str)
    return values[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = _configuration(arguments.config)
    probe_path = resolve_inside(arguments.config.parent, str(config["probe_source"]))
    if probe_path.is_symlink() or not probe_path.is_file():
        raise ValueError("probe source must be a regular file beside the configuration")
    probes = load_jsonl(probe_path, default_role=ProbeRole.VALIDATION, max_probes=10_000)
    if not probes:
        raise ValueError("at least one local generation probe is required")

    # Defense in depth around the adapter's local_files_only=True calls.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    adapter = HuggingFaceCausalLMAdapter()
    started = time.perf_counter()
    model = adapter.load(
        str(arguments.checkpoint),
        device=str(config["device"]),
        dtype=_dtype(config["dtype"]),
    )
    adapter.prepare(model)
    manifest = build_model_manifest(
        model,
        checkpoint=arguments.checkpoint,
        adapter_id=adapter.adapter_id,
    )
    generation = config["generation"]
    assert isinstance(generation, dict)
    policy = GenerationPolicy(
        mode=str(generation["mode"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        temperature=float(generation.get("temperature", 1.0)),
    )
    records: list[dict[str, object]] = []
    for probe in probes:
        sample = adapter.generate(model, adapter.tokenizer().batch([probe.prompt]), policy)[0]
        records.append(
            {
                "finished": sample.finished,
                "output_hash": sha256_bytes(sample.text.encode("utf-8")),
                "probe_id": probe.probe_id,
                "prompt_hash": probe.prompt_hash,
                "token_count": len(sample.token_ids),
                "token_ids_hash": hash_canonical(list(sample.token_ids)),
            }
        )
    payload = {
        "schema_version": 1,
        "runner": "local_huggingface_preflight",
        "runner_status": "EXECUTED",
        "verification_outcome": "NOT_APPLICABLE",
        "benchmark_claims": [],
        "configuration": config,
        "configuration_hash": hash_canonical(config),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
        },
        "manifest": manifest.to_dict(),
        "probe_file_hash": sha256_file(probe_path),
        "probes_hash": probes_hash(probes),
        "generation_records": records,
        "wall_seconds": time.perf_counter() - started,
        "limitations": [
            "This preflight does not compile, compose, extract, or rebase a behavior patch.",
            "No Hugging Face benchmark claim is emitted without a separate full patch workflow.",
        ],
    }
    atomic_write_text(arguments.output, canonical_dumps(payload) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
