"""Patch-factor SafeTensors helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

from modelpact.checkpoints.safetensors import load_safetensors, save_safetensors_atomic


def save_patch_tensors(path: str | Path, tensors: Mapping[str, Tensor]) -> None:
    save_safetensors_atomic(
        path,
        tensors,
        metadata={"format": "modelpact-delta-tensors-v1"},
        overwrite=False,
    )


def load_patch_tensors(path: str | Path) -> dict[str, Tensor]:
    return load_safetensors(path, device="cpu")
