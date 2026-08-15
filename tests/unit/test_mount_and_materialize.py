from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from modelpact.adapters.tiny_lm import (
    TinyCausalLM,
    TinyConfig,
    TinyModelAdapter,
    save_tiny_checkpoint,
)
from modelpact.checkpoints.safetensors import load_safetensors, save_safetensors_atomic
from modelpact.models.schema import inspect_state_schema
from modelpact.patch.ast import (
    Alias,
    DeltaProgram,
    LowRankMatrixDelta,
    SparseMatrixDelta,
    VectorDelta,
)
from modelpact.patch.fold import materialize_patch
from modelpact.patch.mount import mount_patch
from modelpact.util.hashing import sha256_file


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


def linear_patch() -> tuple[DeltaProgram, dict[str, torch.Tensor]]:
    target = "layers.0.mlp.down_proj.weight"
    tensors = {
        "left": torch.full((16, 2), 0.02),
        "right": torch.full((2, 24), -0.03),
    }
    return DeltaProgram({target: LowRankMatrixDelta("left", "right")}), tensors


def test_runtime_mount_changes_output_and_exactly_unmounts() -> None:
    model = tiny_model()
    program, tensors = linear_patch()
    input_ids = torch.tensor([[1, 20, 21]], dtype=torch.long)
    base_parameter = model.layers[0].mlp.down_proj.weight
    base_bytes = base_parameter.detach().clone()
    base_output = model(input_ids).logits.detach().clone()
    session = mount_patch(model, program, tensors)
    assert model.layers[0].mlp.down_proj.weight is not base_parameter
    assert not torch.equal(model(input_ids).logits, base_output)
    with pytest.raises(RuntimeError, match="already mounted"):
        mount_patch(model, program, tensors)
    session.unmount()
    assert model.layers[0].mlp.down_proj.weight is base_parameter
    assert torch.equal(model.layers[0].mlp.down_proj.weight, base_bytes)
    assert torch.equal(model(input_ids).logits, base_output)
    session.unmount()  # idempotent cleanup
    mount_patch(model, program, tensors).unmount()


def test_runtime_mount_gradients_reach_low_rank_factors() -> None:
    model = tiny_model()
    program, tensors = linear_patch()
    session = mount_patch(model, program, tensors, trainable=True)
    model(torch.tensor([[1, 9, 10]])).logits.square().mean().backward()
    factors = session.factor_tensors()
    assert factors["left"].grad is not None
    assert factors["right"].grad is not None
    session.unmount()


def test_tied_runtime_delta_preserves_alias_and_restores_object() -> None:
    model = tiny_model()
    original = model.lm_head.weight
    tensors = {"left": torch.ones(259, 1) * 0.01, "right": torch.ones(1, 16) * 0.01}
    program = DeltaProgram(
        {
            "lm_head.weight": LowRankMatrixDelta("left", "right"),
            "token_embedding.weight": Alias("lm_head.weight"),
        }
    )
    session = mount_patch(model, program, tensors)
    assert torch.equal(model.lm_head.weight, model.token_embedding.weight)
    session.unmount()
    assert model.lm_head.weight is original
    assert model.token_embedding.weight is original


def test_sparse_and_vector_deltas_mount_and_receive_gradients() -> None:
    model = tiny_model()
    program = DeltaProgram(
        {
            "layers.0.mlp.down_proj.weight": SparseMatrixDelta("indices", "values", shape=(16, 24)),
            "final_norm.weight": VectorDelta("norm_delta"),
        }
    )
    tensors = {
        "indices": torch.tensor([[0, 0], [3, 5]], dtype=torch.int64),
        "values": torch.tensor([0.2, -0.1]),
        "norm_delta": torch.full((16,), 0.01),
    }
    session = mount_patch(model, program, tensors, trainable=True)
    model(torch.tensor([[1, 7, 8]])).logits.square().mean().backward()
    assert session.factor_tensors()["values"].grad is not None
    assert session.factor_tensors()["norm_delta"].grad is not None
    session.unmount()


def test_materialization_preserves_source_and_loads_real_patch(tmp_path: Path) -> None:
    model = tiny_model()
    source = save_tiny_checkpoint(model, tmp_path / "source")
    before = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    program, tensors = linear_patch()
    output = tmp_path / "materialized"
    manifest = materialize_patch(
        source,
        output,
        program,
        tensors,
        state_schema=inspect_state_schema(model),
        max_shard_size=5_000,
        patch_ids=("sha256:" + "2" * 64,),
    )
    after = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    assert before == after
    assert (output / "materialization-manifest.json").is_file()
    assert len(manifest["output_files"]) > 1
    for tensor_file in sorted(output.glob("*.safetensors")):
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
            assert handle.metadata() == {
                "format": "pt",
                "modelpact_format": "modelpact-materialized-v1",
            }
    loaded = TinyModelAdapter().load(str(output), device="cpu", dtype=torch.float32)
    target = "layers.0.mlp.down_proj.weight"
    expected = program.apply_to_state(model.state_dict(), tensors)[target]
    assert torch.equal(loaded.state_dict()[target], expected)


def test_materialization_refuses_overwrite_and_nested_output(tmp_path: Path) -> None:
    model = tiny_model()
    source = save_tiny_checkpoint(model, tmp_path / "source")
    program, tensors = linear_patch()
    with pytest.raises(FileExistsError):
        materialize_patch(source, source, program, tensors)
    with pytest.raises(ValueError, match="nested"):
        materialize_patch(source, source / "nested", program, tensors)


def test_materialization_is_byte_deterministic(tmp_path: Path) -> None:
    model = tiny_model()
    source = save_tiny_checkpoint(model, tmp_path / "source")
    program, tensors = linear_patch()
    sharded_source = tmp_path / "sharded-source"
    materialize_patch(source, sharded_source, program, tensors, max_shard_size=5_000)
    source_before = {
        path.name: sha256_file(path) for path in sharded_source.iterdir() if path.is_file()
    }
    first, second = tmp_path / "first", tmp_path / "second"
    manifests = []
    for output in (first, second):
        manifests.append(
            materialize_patch(
                sharded_source,
                output,
                program,
                tensors,
                max_shard_size=5_000,
            )
        )
    source_after = {
        path.name: sha256_file(path) for path in sharded_source.iterdir() if path.is_file()
    }
    assert source_before == source_after
    first_hashes = {
        path.name: sha256_file(path)
        for path in first.iterdir()
        if path.is_file() and path.name != "materialization-manifest.json"
    }
    second_hashes = {
        path.name: sha256_file(path)
        for path in second.iterdir()
        if path.is_file() and path.name != "materialization-manifest.json"
    }
    assert first_hashes == second_hashes
    first_stable = {key: value for key, value in manifests[0].items() if key != "performance"}
    second_stable = {key: value for key, value in manifests[1].items() if key != "performance"}
    assert first_stable == second_stable
    assert len([name for name in manifests[0]["output_files"] if name.endswith(".safetensors")]) > 1
    for manifest in manifests:
        performance = manifest["performance"]
        assert performance["streaming_strategy"] == "planned-output-shard"
        assert performance["read_bytes"] > 0
        assert performance["write_bytes"] > 0
        assert performance["read_seconds"] >= 0
        assert performance["write_seconds"] >= 0
        peak_rss = performance["peak_rss_bytes"]
        if peak_rss is None:
            assert performance["peak_rss_method"] == "unavailable"
        else:
            assert isinstance(peak_rss, int) and peak_rss > 0


def test_materialization_expands_physically_omitted_tied_key(tmp_path: Path) -> None:
    model = tiny_model()
    source = save_tiny_checkpoint(model, tmp_path / "source")
    tensor_path = source / "model.safetensors"
    checkpoint_tensors = load_safetensors(tensor_path)
    del checkpoint_tensors["lm_head.weight"]
    tensor_path.unlink()
    save_safetensors_atomic(tensor_path, checkpoint_tensors, overwrite=False)
    factors = {"left": torch.ones(259, 1) * 0.01, "right": torch.ones(1, 16) * 0.01}
    program = DeltaProgram(
        {
            "lm_head.weight": LowRankMatrixDelta("left", "right"),
            "token_embedding.weight": Alias("lm_head.weight"),
        }
    )
    output = tmp_path / "materialized"
    materialize_patch(
        source,
        output,
        program,
        factors,
        state_schema=inspect_state_schema(model),
    )
    loaded = TinyModelAdapter().load(str(output), device="cpu", dtype=torch.float32)
    assert loaded.lm_head.weight is loaded.token_embedding.weight


def test_materialization_failure_leaves_source_and_output_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tiny_model()
    source = save_tiny_checkpoint(model, tmp_path / "source")
    before = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    program, tensors = linear_patch()
    output = tmp_path / "materialized"
    real_save = save_safetensors_atomic
    calls = 0

    def fail_on_second_shard(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected shard write failure")
        real_save(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "modelpact.checkpoints.writer.save_safetensors_atomic", fail_on_second_shard
    )
    with pytest.raises(RuntimeError, match="injected shard write failure"):
        materialize_patch(source, output, program, tensors, max_shard_size=5_000)

    after = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    assert before == after
    assert not output.exists()
    assert not list(tmp_path.glob(".materialized.*"))
