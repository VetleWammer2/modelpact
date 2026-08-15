from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from modelpact.checkpoints.store import checkpoint_files
from modelpact.checkpoints.writer import _source_file_hashes


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")


def test_direct_checkpoint_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real.safetensors"
    save_file({"weight": torch.ones(1)}, target)
    link = tmp_path / "linked.safetensors"
    _symlink_or_skip(link, target)

    with pytest.raises(ValueError, match="may not be a symlink"):
        checkpoint_files(link)


def test_checkpoint_index_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside-index.json"
    outside.write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")
    _symlink_or_skip(checkpoint / "model.safetensors.index.json", outside)

    with pytest.raises(ValueError, match="regular file"):
        checkpoint_files(checkpoint)


def test_source_auxiliary_size_is_checked_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")
    (checkpoint / "config.json").write_bytes(b"{}")
    monkeypatch.setattr("modelpact.checkpoints.writer.MAX_AUXILIARY_FILE_BYTES", 1)

    def forbidden_hash(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized checkpoint file was hashed")

    monkeypatch.setattr("modelpact.checkpoints.writer.sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="exceeds size limit"):
        _source_file_hashes(checkpoint)
