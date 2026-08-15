from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from modelpact.adapters.base import GenerationPolicy
from modelpact.adapters.tiny_lm import (
    TinyCausalLM,
    TinyConfig,
    TinyModelAdapter,
    TinyTokenizer,
    TinyTrainingConfig,
    save_tiny_checkpoint,
    train_tiny_causal_lm,
)
from modelpact.checkpoints.safetensors import load_safetensors, save_safetensors_atomic
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.models.schema import ModelStateSchema, inspect_state_schema


def tiny_config(*, seed: int = 17) -> TinyConfig:
    return TinyConfig(
        max_sequence_length=32,
        hidden_size=16,
        intermediate_size=24,
        num_layers=1,
        num_heads=4,
        initialization_seed=seed,
    )


def test_tiny_tokenizer_utf8_roundtrip_and_batch() -> None:
    tokenizer = TinyTokenizer()
    text = "på pact ✓"
    encoded = tokenizer.encode(text, add_eos=True)
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(encoded) == text
    batch = tokenizer.batch(["a", "longer"])
    assert batch.input_ids.shape == batch.attention_mask.shape == (2, 7)
    assert batch.attention_mask[0].sum() == 2


def test_tiny_initialization_is_deterministic_without_changing_global_rng() -> None:
    torch.manual_seed(901)
    expected = torch.rand(3)
    torch.manual_seed(901)
    first = TinyCausalLM(tiny_config())
    observed = torch.rand(3)
    second = TinyCausalLM(tiny_config())
    assert torch.equal(expected, observed)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_tiny_checkpoint_roundtrip_and_tied_alias(tmp_path: Path) -> None:
    model = TinyCausalLM(tiny_config())
    checkpoint = save_tiny_checkpoint(model, tmp_path / "tiny")
    loaded = TinyModelAdapter().load(str(checkpoint), device="cpu", dtype=torch.float32)
    assert loaded.lm_head.weight is loaded.token_embedding.weight
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_tiny_adapter_rejects_inconsistent_tied_checkpoint(tmp_path: Path) -> None:
    model = TinyCausalLM(tiny_config())
    checkpoint = save_tiny_checkpoint(model, tmp_path / "tiny")
    tensor_file = checkpoint / "model.safetensors"
    tensors = load_safetensors(tensor_file)
    tensors["lm_head.weight"] = tensors["lm_head.weight"] + 1
    tensor_file.unlink()
    save_safetensors_atomic(tensor_file, tensors, overwrite=False)
    with pytest.raises(ValueError, match="tied embedding alias"):
        TinyModelAdapter().load(str(checkpoint), device="cpu", dtype=torch.float32)


def test_tiny_real_forward_and_generation_are_deterministic() -> None:
    adapter = TinyModelAdapter()
    model = TinyCausalLM(tiny_config())
    batch = adapter.tokenizer().batch(["abc", "d"])
    logits = adapter.forward_logits(model, batch)
    assert logits.shape == (2, 4, adapter.tokenizer().vocab_size)
    policy = GenerationPolicy(mode="sample", max_new_tokens=3, seed=81)
    first = adapter.generate(model, batch, policy)
    second = adapter.generate(model, batch, policy)
    assert first == second
    assert all(len(sample.token_ids) <= 3 for sample in first)


def test_tiny_training_executes_deterministically() -> None:
    left = TinyCausalLM(tiny_config())
    right = TinyCausalLM(tiny_config())
    configuration = TinyTrainingConfig(steps=3, batch_size=2, seed=55)
    corpus = ["alpha -> one", "beta -> two", "gamma -> three"]
    left_losses = train_tiny_causal_lm(left, corpus, config=configuration)
    right_losses = train_tiny_causal_lm(right, corpus, config=configuration)
    assert left_losses == right_losses
    assert len(left_losses) == configuration.steps
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters(), strict=True):
        assert torch.equal(left_parameter, right_parameter)


def test_state_schema_records_patchable_state_and_physical_alias() -> None:
    model = TinyCausalLM(tiny_config())
    schema = inspect_state_schema(model)
    assert schema.tensor("layers.0.mlp.down_proj.weight").patchable
    assert schema.tensor("final_norm.weight").kind == "norm_scale"
    assert schema.aliases[0].members == ("lm_head.weight", "token_embedding.weight")
    assert ModelStateSchema.from_dict(schema.to_dict()) == schema


def test_model_manifest_is_stable_and_covers_tokenizer(tmp_path: Path) -> None:
    model = TinyCausalLM(tiny_config())
    checkpoint = save_tiny_checkpoint(model, tmp_path / "tiny")
    first = build_model_manifest(
        model,
        checkpoint=checkpoint,
        adapter_id=TinyModelAdapter.adapter_id,
        architecture_config=model.config.to_dict(),
    )
    second = build_model_manifest(
        model,
        checkpoint=checkpoint,
        adapter_id=TinyModelAdapter.adapter_id,
        architecture_config=model.config.to_dict(),
    )
    assert first.manifest_hash == second.manifest_hash
    assert ModelManifest.from_dict(first.to_dict()) == first
    tokenizer_path = checkpoint / "tokenizer.json"
    tokenizer_value = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_value["test_mutation"] = True
    tokenizer_path.write_text(json.dumps(tokenizer_value), encoding="utf-8")
    changed = build_model_manifest(
        model,
        checkpoint=checkpoint,
        adapter_id=TinyModelAdapter.adapter_id,
        architecture_config=model.config.to_dict(),
    )
    assert first.signature.tokenizer_hash != changed.signature.tokenizer_hash
    assert first.signature.checkpoint_hash == changed.signature.checkpoint_hash
