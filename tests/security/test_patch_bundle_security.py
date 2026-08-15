from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from modelpact.adapters.tiny_lm import TinyCausalLM, TinyConfig
from modelpact.models.schema import inspect_state_schema
from modelpact.patch.ast import DeltaProgram, LowRankMatrixDelta
from modelpact.patch.bundle import create_patch_bundle, load_patch_bundle
from modelpact.patch.validate import load_delta_program
from modelpact.util.canonical_json import canonical_dumps


def test_delta_program_rejects_path_traversal_tensor_reference(tmp_path: Path) -> None:
    value = {
        "schema_version": 1,
        "targets": {
            "layer.weight": {
                "op": "low_rank_matrix",
                "left": "../outside",
                "right": "right",
                "scale": 1,
            }
        },
    }
    path = tmp_path / "program.json"
    path.write_text(canonical_dumps(value), encoding="utf-8")
    program = load_delta_program(path)
    with pytest.raises(ValueError, match="unsafe tensor reference"):
        program.validate({"../outside": torch.ones(2, 1), "right": torch.ones(1, 2)})


def test_bundle_rejects_manifest_artifact_traversal(tmp_path: Path) -> None:
    model = TinyCausalLM(
        TinyConfig(
            max_sequence_length=16,
            hidden_size=8,
            intermediate_size=12,
            num_layers=1,
            num_heads=2,
        )
    )
    schema = inspect_state_schema(model)
    tensors = {"left": torch.ones(8, 1), "right": torch.ones(1, 12)}
    program = DeltaProgram({"layers.0.mlp.down_proj.weight": LowRankMatrixDelta("left", "right")})
    bundle = create_patch_bundle(
        tmp_path / "patch",
        name="secure",
        base_signature={
            "schema_version": 1,
            "adapter_id": "test.adapter",
            "architecture_hash": "sha256:" + "0" * 64,
            "state_schema_hash": schema.schema_hash,
            "checkpoint_hash": "sha256:" + "1" * 64,
            "tokenizer_hash": "sha256:" + "2" * 64,
            "chat_template_hash": "sha256:" + "3" * 64,
            "generation_config_hash": "sha256:" + "4" * 64,
        },
        state_schema=schema,
        program=program,
        tensors=tensors,
        tool_version="0.1.0",
    )
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["../outside"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe artifact path"):
        load_patch_bundle(bundle.path)


def test_bundle_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="regular directory"):
        load_patch_bundle(link)
