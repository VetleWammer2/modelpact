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

from modelpact.util.canonical_json import canonical_json_bytes, strict_json_loads
from modelpact.util.hashing import sha256_bytes

DEFAULT_MAX_FILE_BYTES = 16 * 1024**3
DEFAULT_MAX_TENSORS = 100_000
DEFAULT_MAX_TENSOR_ELEMENTS = 1 << 40
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024**2


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


def _canonicalize_safetensors_header(path: Path) -> None:
    """Canonicalize a freshly written SafeTensors header in place.

    The SafeTensors serializer does not promise deterministic map iteration for
    metadata. Preserve its allocated header length and tensor payload offsets,
    but replace the generated JSON with ModelPact's canonical encoding.
    """

    file_size = path.stat().st_size
    with path.open("r+b") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError("freshly written SafeTensors file lacks a complete header")
        header_size = int.from_bytes(prefix, byteorder="little", signed=False)
        if (
            header_size <= 0
            or header_size > MAX_SAFETENSORS_HEADER_BYTES
            or header_size > file_size - 8
        ):
            raise RuntimeError("freshly written SafeTensors file has an invalid header size")
        encoded_header = handle.read(header_size)
        try:
            header = strict_json_loads(encoded_header)
        except ValueError as error:
            raise RuntimeError(
                "freshly written SafeTensors file has invalid header JSON"
            ) from error
        if not isinstance(header, dict):
            raise RuntimeError("freshly written SafeTensors header must be an object")
        canonical = canonical_json_bytes(header)
        if len(canonical) > header_size:
            raise RuntimeError("canonical SafeTensors header exceeds its allocated size")
        handle.seek(8)
        handle.write(canonical)
        handle.write(b" " * (header_size - len(canonical)))
        handle.flush()
        os.fsync(handle.fileno())


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
        _canonicalize_safetensors_header(temporary)
        if not overwrite and target.exists():
            raise FileExistsError(target)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
