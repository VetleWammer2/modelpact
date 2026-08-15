from __future__ import annotations

from pathlib import Path

import pytest
import torch

from modelpact.adapters.tiny_lm import TinyCausalLM, TinyConfig
from modelpact.contracts.parser import parse_contract
from modelpact.models.schema import inspect_state_schema
from modelpact.patch.ast import (
    Alias,
    DeltaProgram,
    LowRankMatrixDelta,
    SparseMatrixDelta,
    Sum,
    VectorDelta,
)
from modelpact.patch.bundle import (
    attach_bundle_artifacts,
    create_patch_bundle,
    load_patch_bundle,
    missing_bundle_artifacts,
)
from modelpact.patch.validate import validate_base_signature
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical


def tiny_model() -> TinyCausalLM:
    return TinyCausalLM(
        TinyConfig(
            max_sequence_length=32,
            hidden_size=16,
            intermediate_size=24,
            num_layers=1,
            num_heads=4,
        )
    )


def base_signature(schema_hash: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "adapter_id": "test.adapter",
        "architecture_hash": "sha256:" + "0" * 64,
        "state_schema_hash": schema_hash,
        "checkpoint_hash": "sha256:" + "1" * 64,
        "tokenizer_hash": "sha256:" + "2" * 64,
        "chat_template_hash": "sha256:" + "3" * 64,
        "generation_config_hash": "sha256:" + "4" * 64,
    }


def test_low_rank_sum_sparse_and_vector_semantics() -> None:
    low_rank = LowRankMatrixDelta("left", "right", scale=0.5)
    sparse = SparseMatrixDelta("indices", "values", shape=(2, 3))
    tensors = {
        "left": torch.tensor([[1.0], [2.0]]),
        "right": torch.tensor([[3.0, 4.0, 5.0]]),
        "indices": torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
        "values": torch.tensor([7.0, 11.0]),
    }
    operation = Sum((low_rank, sparse))
    expected = 0.5 * tensors["left"] @ tensors["right"]
    expected[0, 1] += 7
    expected[1, 2] += 11
    assert torch.equal(operation.materialize(tensors), expected)
    vector = VectorDelta("vector", 2)
    assert torch.equal(
        vector.apply(torch.ones(3), {"vector": torch.arange(3.0)}), torch.tensor([1.0, 3.0, 5.0])
    )


def test_delta_program_serialization_and_additive_application() -> None:
    state = {"layer.weight": torch.zeros(2, 3), "unchanged": torch.ones(1)}
    tensors = {"left": torch.ones(2, 1), "right": torch.tensor([[1.0, 2.0, 3.0]])}
    program = DeltaProgram({"layer.weight": LowRankMatrixDelta("left", "right")})
    parsed = DeltaProgram.from_dict(program.to_dict())
    result = parsed.apply_to_state(state, tensors)
    assert torch.equal(result["layer.weight"], tensors["left"] @ tensors["right"])
    assert torch.equal(result["unchanged"], state["unchanged"])
    assert result["unchanged"] is not state["unchanged"]


def test_sparse_rejects_duplicate_and_out_of_bounds_indices() -> None:
    operation = SparseMatrixDelta("indices", "values", shape=(2, 2))
    with pytest.raises(ValueError, match="sorted and unique"):
        operation.validate(
            {
                "indices": torch.tensor([[0, 0], [0, 0]]),
                "values": torch.ones(2),
            }
        )
    with pytest.raises(ValueError, match="out of bounds"):
        operation.validate({"indices": torch.tensor([[2, 0]]), "values": torch.ones(1)})


def test_low_rank_rejects_shape_and_dtype_errors() -> None:
    operation = LowRankMatrixDelta("left", "right")
    with pytest.raises(ValueError, match="shapes"):
        operation.validate({"left": torch.ones(2, 2), "right": torch.ones(3, 4)})
    with pytest.raises(ValueError, match="floating dtype"):
        operation.validate(
            {
                "left": torch.ones(2, 1, dtype=torch.long),
                "right": torch.ones(1, 3, dtype=torch.long),
            }
        )


def test_alias_cycle_and_unknown_operation_are_rejected() -> None:
    cyclic = DeltaProgram({"a": Alias("b"), "b": Alias("a")})
    with pytest.raises(ValueError, match="cycle"):
        cyclic.validate({})
    with pytest.raises(ValueError, match="unknown delta operation"):
        DeltaProgram.from_dict(
            {"schema_version": 1, "targets": {"x": {"op": "execute_python", "code": "1+1"}}}
        )


def test_tied_alias_requires_complete_consistent_delta() -> None:
    model = tiny_model()
    schema = inspect_state_schema(model)
    tensors = {"left": torch.ones(259, 1), "right": torch.ones(1, 16)}
    partial = DeltaProgram({"lm_head.weight": LowRankMatrixDelta("left", "right")})
    with pytest.raises(ValueError, match="omits aliases"):
        partial.validate(tensors, schema)
    complete = DeltaProgram(
        {
            "lm_head.weight": LowRankMatrixDelta("left", "right"),
            "token_embedding.weight": Alias("lm_head.weight"),
        }
    )
    complete.validate(tensors, schema)


def test_patch_bundle_is_stable_and_detects_tensor_mutation(tmp_path: Path) -> None:
    model = tiny_model()
    schema = inspect_state_schema(model)
    target = "layers.0.mlp.down_proj.weight"
    tensors = {"left": torch.ones(16, 1), "right": torch.ones(1, 24)}
    program = DeltaProgram({target: LowRankMatrixDelta("left", "right")})
    bundle = create_patch_bundle(
        tmp_path / "patch",
        name="test-patch",
        base_signature=base_signature(schema.schema_hash),
        state_schema=schema,
        program=program,
        tensors=tensors,
        tool_version="0.1.0",
        compiler_configuration={"rank": 1},
    )
    assert bundle.manifest.patch_id == hash_canonical(bundle.manifest.identity_payload())
    assert (
        load_patch_bundle(bundle.path, state_schema=schema).manifest.patch_id
        == bundle.manifest.patch_id
    )
    tensor_path = bundle.path / "tensors.safetensors"
    data = bytearray(tensor_path.read_bytes())
    data[-1] ^= 1
    tensor_path.write_bytes(data)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_patch_bundle(bundle.path, state_schema=schema)


def test_evidence_can_be_attached_without_changing_patch_identity(tmp_path: Path) -> None:
    model = tiny_model()
    schema = inspect_state_schema(model)
    tensors = {"left": torch.ones(16, 1), "right": torch.ones(1, 24)}
    program = DeltaProgram({"layers.0.mlp.down_proj.weight": LowRankMatrixDelta("left", "right")})
    bundle = create_patch_bundle(
        tmp_path / "patch",
        name="evidence-test",
        base_signature=base_signature(schema.schema_hash),
        state_schema=schema,
        program=program,
        tensors=tensors,
        tool_version="0.1.0",
    )
    patch_id = bundle.manifest.patch_id
    evidence_id = bundle.evidence_id
    attached = attach_bundle_artifacts(
        bundle.path,
        {
            "evidence/compile.json": b"{}",
            "certificate.json": b'{"outcome":"INCONCLUSIVE"}',
            "apply_patch.py": b"# generated standalone helper\n",
        },
        state_schema=schema,
    )
    assert attached.manifest.patch_id == patch_id
    assert attached.evidence_id != evidence_id
    assert attached.bundle_id != bundle.bundle_id
    assert "evidence/compile.json" in attached.manifest.artifact_hashes
    assert "evidence/validation.json" in missing_bundle_artifacts(attached.manifest)


def test_apply_time_base_signature_comparison_is_exact() -> None:
    signature = {
        "adapter_id": "adapter",
        "architecture_hash": "sha256:a",
        "state_schema_hash": "sha256:b",
        "checkpoint_hash": "sha256:c",
        "tokenizer_hash": "sha256:d",
        "chat_template_hash": "sha256:e",
        "generation_config_hash": "sha256:f",
    }
    validate_base_signature(signature, dict(signature))
    changed = {**signature, "tokenizer_hash": "sha256:changed"}
    with pytest.raises(ValueError, match="tokenizer_hash"):
        validate_base_signature(signature, changed)


def test_contract_content_changes_patch_identity(tmp_path: Path) -> None:
    model = tiny_model()
    schema = inspect_state_schema(model)
    tensors = {"left": torch.ones(16, 1), "right": torch.ones(1, 24)}
    program = DeltaProgram({"layers.0.mlp.down_proj.weight": LowRankMatrixDelta("left", "right")})
    patch_ids = []
    for index, threshold in enumerate((0.1, 0.2)):
        contract = parse_contract(
            {
                "compile": {"objectives": []},
                "contract_version": 1,
                "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
                "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
                "id": "contract-identity",
                "model_requirements": {"output_semantics": "causal_lm"},
                "schema_version": 1,
                "statistics": {
                    "bootstrap_samples": 10,
                    "bootstrap_seed": 1,
                    "confidence_level": 0.95,
                },
                "verify": {
                    "guards": [],
                    "targets": [
                        {
                            "id": "bounded-drift",
                            "maximum_mean": threshold,
                            "source": "probes.jsonl",
                            "type": "base_kl",
                        }
                    ],
                },
            }
        )
        bundle = create_patch_bundle(
            tmp_path / f"patch-{index}",
            name="contract-identity",
            base_signature=base_signature(schema.schema_hash),
            state_schema=schema,
            program=program,
            tensors=tensors,
            tool_version="0.1.0",
            contracts={
                "contracts/target.yaml": (canonical_dumps(contract.to_dict()) + "\n").encode()
            },
            provides=(contract.contract_id,),
        )
        patch_ids.append(bundle.manifest.patch_id)
    assert patch_ids[0] != patch_ids[1]
