"""Bounded SafeTensors I/O with deterministic tensor ordering."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from modelpact.util.hashing import sha256_bytes

DEFAULT_MAX_FILE_BYTES = 16 * 1024**3
DEFAULT_MAX_TENSORS = 100_000
DEFAULT_MAX_TENSOR_ELEMENTS = 1 << 40


def _plain_file(path: Path, *, max_file_bytes: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular SafeTensors file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_file_bytes:
        raise ValueError(f"invalid SafeTensors file size ({size} bytes): {path}")


def tensor_content_hash(tensor: Tensor) -> str:
    """Hash dtype, shape, and exact CPU tensor bytes."""

    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    prefix = f"{value.dtype}|{tuple(value.shape)}|".encode()
    return sha256_bytes(prefix + raw)


def load_safetensors(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    max_tensor_elements: int = DEFAULT_MAX_TENSOR_ELEMENTS,
) -> dict[str, Tensor]:
    """Load one data-only tensor file after bounded metadata validation."""

    source = Path(path)
    _plain_file(source, max_file_bytes=max_file_bytes)
    result: dict[str, Tensor] = {}
    # SafeTensors currently ships no complete typing for safe_open.
    with safe_open(source, framework="pt", device=str(device)) as handle:  # type: ignore[no-untyped-call]
        keys = sorted(handle.keys())
        if len(keys) > max_tensors:
            raise ValueError(f"SafeTensors key count exceeds {max_tensors}")
        for key in keys:
            if not key or len(key) > 2048 or "\x00" in key:
                raise ValueError("invalid tensor key")
            tensor = handle.get_tensor(key)
            if tensor.numel() > max_tensor_elements:
                raise ValueError(f"tensor exceeds element bound: {key}")
            # Detach from the file mapping so bundle files can be independently
            # re-hashed, moved, or removed on Windows after this function returns.
            result[key] = tensor.clone()
    return result


def _normalized_tensors(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not tensors:
        raise ValueError("at least one tensor is required")
    normalized: dict[str, Tensor] = {}
    for key in sorted(tensors):
        if not isinstance(key, str) or not key or len(key) > 2048 or "\x00" in key:
            raise ValueError("invalid tensor key")
        value = tensors[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"not a tensor: {key}")
        if value.layout != torch.strided:
            raise ValueError(f"unsupported tensor layout for {key}: {value.layout}")
        # Cloning prevents SafeTensors' shared-storage rejection and ensures a
        # snapshot cannot change underneath the writer.
        normalized[key] = value.detach().to(device="cpu").contiguous().clone()
    return normalized


def save_safetensors_atomic(
    path: str | Path,
    tensors: Mapping[str, Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
    overwrite: bool = True,
) -> None:
    """Write a deterministic SafeTensors file through a same-directory rename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("refusing to replace a symlink")
    if not overwrite and target.exists():
        raise FileExistsError(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".safetensors", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            _normalized_tensors(tensors),
            temporary,
            metadata=dict(sorted((metadata or {}).items())),
        )
        if not overwrite and target.exists():
            raise FileExistsError(target)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
