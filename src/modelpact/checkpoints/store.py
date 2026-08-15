"""Safe discovery of local SafeTensors checkpoint shards."""

from __future__ import annotations

import json
from pathlib import Path

from torch import Tensor

from modelpact.checkpoints.safetensors import load_safetensors

MAX_INDEX_BYTES = 16 * 1024**2


def _safe_child(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError(f"unsafe checkpoint path: {relative}")
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise ValueError(f"checkpoint shard may not be a symlink: {relative}")
    resolved_root = root.resolve()
    resolved = unresolved.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(f"checkpoint path escapes its directory: {relative}")
    return resolved


def checkpoint_files(checkpoint: str | Path) -> tuple[Path, ...]:
    source = Path(checkpoint)
    if source.is_file():
        if source.suffix != ".safetensors":
            raise ValueError("checkpoint file must use SafeTensors")
        return (source,)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {source}")
    index = source / "model.safetensors.index.json"
    if index.exists():
        if index.stat().st_size > MAX_INDEX_BYTES:
            raise ValueError("checkpoint index exceeds size limit")
        value = json.loads(index.read_text(encoding="utf-8"))
        weight_map = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("malformed SafeTensors checkpoint index")
        if not all(
            isinstance(key, str) and isinstance(item, str) for key, item in weight_map.items()
        ):
            raise ValueError("malformed checkpoint weight map")
        files = tuple(sorted({_safe_child(source, item) for item in weight_map.values()}))
    else:
        files = tuple(sorted(source.glob("*.safetensors")))
    if not files:
        raise ValueError(f"no SafeTensors files found in {source}")
    if any(file.is_symlink() for file in files):
        raise ValueError("checkpoint shards may not be symlinks")
    return files


def load_checkpoint_tensors(checkpoint: str | Path, *, device: str = "cpu") -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for shard in checkpoint_files(checkpoint):
        for key, tensor in load_safetensors(shard, device=device).items():
            if key in result:
                raise ValueError(f"duplicate tensor key across checkpoint shards: {key}")
            result[key] = tensor
    return dict(sorted(result.items()))
