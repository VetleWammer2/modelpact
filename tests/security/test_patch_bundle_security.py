from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from modelpact.adapters.tiny_lm import TinyCausalLM, TinyConfig
from modelpact.contracts.parser import parse_contract
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


def test_untrusted_patch_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    program = tmp_path / "program.json"
    program.write_text(
        '{"schema_version":1,"schema_version":1,"targets":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed delta program JSON"):
        load_delta_program(program)


def test_bundle_manifest_requires_exact_sha256_digests(tmp_path: Path) -> None:
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
    bundle = create_patch_bundle(
        tmp_path / "patch",
        name="digest-bounds",
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
        program=DeltaProgram(
            {"layers.0.mlp.down_proj.weight": LowRankMatrixDelta("left", "right")}
        ),
        tensors={"left": torch.ones(8, 1), "right": torch.ones(1, 12)},
        tool_version="0.1.0",
    )
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["delta-program.json"] = "sha256:short"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid artifact digest"):
        load_patch_bundle(bundle.path)


def test_bundle_manifest_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "patch"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed patch manifest JSON"):
        load_patch_bundle(root)


def test_bundle_rejects_oversized_artifact_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    bundle = create_patch_bundle(
        tmp_path / "patch",
        name="bounded-read",
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
        program=DeltaProgram(
            {"layers.0.mlp.down_proj.weight": LowRankMatrixDelta("left", "right")}
        ),
        tensors={"left": torch.ones(8, 1), "right": torch.ones(1, 12)},
        tool_version="0.1.0",
    )
    evidence = bundle.path / "evidence" / "oversized.json"
    evidence.parent.mkdir()
    evidence.write_bytes(b"{}")
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["evidence/oversized.json"] = "sha256:" + "5" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr("modelpact.patch.bundle.MAX_BUNDLE_ARTIFACT_BYTES", 1)

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized bundle artifact was hashed")

    monkeypatch.setattr("modelpact.patch.bundle.sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="exceeds size limit"):
        load_patch_bundle(bundle.path)


def test_bundle_claims_must_bind_embedded_executable_contracts(tmp_path: Path) -> None:
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
    contract = parse_contract(
        {
            "compile": {"objectives": []},
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": "bound-claim",
            "model_requirements": {"output_semantics": "causal_lm"},
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 10,
                "bootstrap_seed": 1,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [
                    {
                        "id": "guard",
                        "maximum_mean": 1.0,
                        "source": "guards.jsonl",
                        "type": "base_kl",
                    }
                ],
                "targets": [
                    {
                        "id": "target",
                        "maximum": 1,
                        "minimum": 1,
                        "source": "targets.jsonl",
                        "type": "generation_length",
                    }
                ],
            },
        }
    )
    output = tmp_path / "unbound"
    with pytest.raises(ValueError, match="provides claims do not match"):
        create_patch_bundle(
            output,
            name="unbound",
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
            program=DeltaProgram(
                {"layers.0.mlp.down_proj.weight": LowRankMatrixDelta("left", "right")}
            ),
            tensors={"left": torch.ones(8, 1), "right": torch.ones(1, 12)},
            tool_version="0.1.0",
            contracts={
                "contracts/target.yaml": (canonical_dumps(contract.to_dict()) + "\n").encode()
            },
            provides=("sha256:" + "f" * 64,),
            preserves=(contract.contract_id,),
        )
    assert not output.exists()
