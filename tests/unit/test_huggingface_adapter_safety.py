from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from modelpact.adapters.base import GenerationPolicy
from modelpact.adapters.huggingface import (
    HuggingFaceCausalLMAdapter,
    HuggingFaceTokenizerAdapter,
)


class _FakeTokenizer:
    pad_token_id = None
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 32

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [3 + ord(character) % 20 for character in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return ",".join(map(str, token_ids))


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.config = SimpleNamespace(is_decoder=True, use_cache=True)


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    save_file(
        {"projection.weight": torch.ones((1, 1))},
        root / "model.safetensors",
        metadata={"format": "pt"},
    )
    return root


def test_huggingface_load_forces_local_safe_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path / "hf")
    calls: dict[str, dict[str, object]] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _FakeTokenizer:
            assert path == checkpoint
            calls["tokenizer"] = kwargs
            return _FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _FakeModel:
            assert path == checkpoint
            calls["model"] = kwargs
            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    adapter = HuggingFaceCausalLMAdapter()
    model = adapter.load(str(checkpoint), device="cpu", dtype=torch.float32)
    assert isinstance(model, _FakeModel)
    assert calls["tokenizer"] == {"local_files_only": True, "trust_remote_code": False}
    assert calls["model"] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "torch_dtype": torch.float32,
    }


def test_huggingface_accepts_causal_config_with_is_decoder_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path / "hf")
    model = _FakeModel()
    model.config.is_decoder = False
    model.config.is_encoder_decoder = False

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _FakeTokenizer:
            assert path == checkpoint
            assert kwargs["local_files_only"] is True
            return _FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _FakeModel:
            assert path == checkpoint
            assert kwargs["trust_remote_code"] is False
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded = HuggingFaceCausalLMAdapter().load(str(checkpoint), device="cpu", dtype=torch.float32)
    assert loaded is model


def test_huggingface_tokenizer_uses_eos_fallback_and_left_padding() -> None:
    tokenizer = HuggingFaceTokenizerAdapter(_FakeTokenizer())
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    batch = tokenizer.batch(["a", "abc"])
    assert batch.input_ids[0, 0].item() == tokenizer.pad_token_id
    assert not batch.attention_mask[0, 0]
    assert batch.attention_mask[1].all()


def test_huggingface_adapter_rejects_nonlocal_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local regular directory"):
        HuggingFaceCausalLMAdapter().load(
            str(tmp_path / "missing"), device="cpu", dtype=torch.float32
        )


def test_huggingface_adapter_rejects_symlinked_checkpoint(tmp_path: Path) -> None:
    target = _checkpoint(tmp_path / "target")
    link = tmp_path / "linked-checkpoint"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")

    with pytest.raises(ValueError, match="local regular directory"):
        HuggingFaceCausalLMAdapter().load(str(link))


def test_huggingface_preflight_rejects_pickle_before_transformers_loaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path / "hf")
    (checkpoint / "pytorch_model.bin").write_bytes(b"pickle must never be inspected")
    calls = {"model": 0, "tokenizer": 0}

    class BombTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            calls["tokenizer"] += 1
            raise AssertionError("tokenizer loader must not run")

    class BombModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            calls["model"] += 1
            raise AssertionError("model loader must not run")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = BombTokenizer
    fake_transformers.AutoModelForCausalLM = BombModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(ValueError, match="non-SafeTensors"):
        HuggingFaceCausalLMAdapter().load(str(checkpoint))
    assert calls == {"model": 0, "tokenizer": 0}


def test_huggingface_preflight_rejects_traversing_shard_before_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.safetensors"
    save_file({"weight": torch.ones(1)}, outside)
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"../outside.safetensors"}}\n',
        encoding="utf-8",
    )
    called = False

    class BombLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            nonlocal called
            called = True
            raise AssertionError("Transformers loader must not run")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = BombLoader
    fake_transformers.AutoModelForCausalLM = BombLoader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(ValueError, match="unsafe checkpoint path"):
        HuggingFaceCausalLMAdapter().load(str(checkpoint))
    assert called is False


def test_huggingface_preflight_bounds_config_and_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import modelpact.adapters.huggingface as huggingface_module
    import modelpact.checkpoints.store as checkpoint_store

    called = False

    class BombLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            nonlocal called
            called = True
            raise AssertionError("Transformers loader must not run")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = BombLoader
    fake_transformers.AutoModelForCausalLM = BombLoader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    checkpoint = _checkpoint(tmp_path / "config-too-large")
    monkeypatch.setattr(huggingface_module, "_MAX_CONFIG_BYTES", 1)
    with pytest.raises(ValueError, match="configuration exceeds"):
        HuggingFaceCausalLMAdapter().load(str(checkpoint))

    indexed = tmp_path / "index-too-large"
    indexed.mkdir()
    (indexed / "config.json").write_text("{}\n", encoding="utf-8")
    (indexed / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(huggingface_module, "_MAX_CONFIG_BYTES", 16 * 1024**2)
    monkeypatch.setattr(checkpoint_store, "MAX_INDEX_BYTES", 1)
    with pytest.raises(ValueError, match="index exceeds"):
        HuggingFaceCausalLMAdapter().load(str(indexed))
    assert called is False


def test_huggingface_generation_honors_sampling_policy_and_restores_rng(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path / "hf")
    calls: list[dict[str, object]] = []

    class GeneratingModel(_FakeModel):
        def generate(self, **kwargs: object) -> torch.Tensor:
            calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            generated = torch.randint(3, 30, (input_ids.shape[0], 1))
            return torch.cat((input_ids, generated), dim=1)

    model = GeneratingModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> _FakeTokenizer:
            return _FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> GeneratingModel:
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    adapter = HuggingFaceCausalLMAdapter()
    loaded = adapter.load(str(checkpoint))
    batch = adapter.tokenizer().batch(("prompt",))
    policy = GenerationPolicy(
        mode="sample",
        max_new_tokens=1,
        seed=712,
        temperature=0.7,
        top_k=9,
        top_p=0.8,
    )
    torch.manual_seed(81)
    state = torch.random.get_rng_state().clone()
    first = adapter.generate(loaded, batch, policy)
    second = adapter.generate(loaded, batch, policy)

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), state)
    assert len(calls) == 2
    for call in calls:
        assert set(call) == {
            "attention_mask",
            "do_sample",
            "eos_token_id",
            "input_ids",
            "max_new_tokens",
            "pad_token_id",
            "temperature",
            "top_k",
            "top_p",
            "use_cache",
        }
        assert call["do_sample"] is True
        assert call["temperature"] == 0.7
        assert call["top_k"] == 9
        assert call["top_p"] == 0.8
        assert call["use_cache"] is False
