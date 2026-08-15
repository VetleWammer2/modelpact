from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

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


def test_huggingface_load_forces_local_safe_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
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
