from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from modelpact.codegen._huggingface_adapter_runtime import (
    StandaloneHuggingFaceModelAdapter,
)


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(is_encoder_decoder=False, use_cache=True)


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    save_file(
        {"weight": torch.ones(1)},
        root / "model.safetensors",
        metadata={"format": "pt"},
    )
    return root


def test_standalone_huggingface_loader_forces_local_safe_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    calls: dict[str, dict[str, object]] = {}
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            calls["tokenizer"] = {"path": path, **kwargs}
            return object()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> nn.Module:
            calls["model"] = {"path": path, **kwargs}
            return _FakeModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizerLoader  # type: ignore[attr-defined]
    fake_transformers.AutoModelForCausalLM = FakeModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = StandaloneHuggingFaceModelAdapter().load(str(checkpoint))

    assert isinstance(model, _FakeModel)
    for call in calls.values():
        assert call["path"] == str(checkpoint.resolve())
        assert call["local_files_only"] is True
        assert call["trust_remote_code"] is False
    assert calls["model"]["use_safetensors"] is True
    assert calls["model"]["torch_dtype"] is torch.float32
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_standalone_huggingface_loader_rejects_unsafe_weight_files(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "pytorch_model.bin").write_bytes(b"not a trusted weight format")

    with pytest.raises(ValueError, match="non-SafeTensors"):
        StandaloneHuggingFaceModelAdapter().load(str(checkpoint))


def test_standalone_huggingface_loader_rejects_symlinked_checkpoint(
    tmp_path: Path,
) -> None:
    target = _checkpoint(tmp_path / "target")
    link = tmp_path / "checkpoint-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")

    with pytest.raises(ValueError, match="local regular directory"):
        StandaloneHuggingFaceModelAdapter().load(str(link))
