"""Content fingerprints for local checkpoints and model-side configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from modelpact.checkpoints.safetensors import DEFAULT_MAX_FILE_BYTES, tensor_content_hash
from modelpact.checkpoints.store import checkpoint_files
from modelpact.util.canonical_json import strict_json_loads
from modelpact.util.hashing import hash_canonical, sha256_file

MAX_CONFIG_BYTES = 16 * 1024**2
MAX_TOKENIZER_FILE_BYTES = 2 * 1024**3
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "tokenizer.model",
)


def _bounded_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular configuration file: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(f"configuration file exceeds size limit: {path}")
    return strict_json_loads(path.read_bytes())


def canonical_config(checkpoint: str | Path) -> Mapping[str, object]:
    root = Path(checkpoint)
    if root.is_file():
        return {}
    path = root / "config.json"
    if not path.exists():
        return {}
    value = _bounded_json(path)
    if not isinstance(value, dict):
        raise ValueError("model config must be a JSON object")
    # Location and private cache details are not architecture semantics.
    excluded = {"_name_or_path", "transformers_version", "torch_dtype"}
    return {key: value[key] for key in sorted(value) if key not in excluded}


def configuration_fingerprint(checkpoint: str | Path) -> str:
    """Fingerprint the canonical, location-independent model configuration."""

    return hash_canonical(canonical_config(checkpoint))


def checkpoint_tensor_fingerprint(checkpoint: str | Path) -> tuple[str, dict[str, str]]:
    """Hash every checkpoint tensor rather than relying on a path or model name."""

    from safetensors import safe_open

    hashes: dict[str, str] = {}
    for shard in checkpoint_files(checkpoint):
        if shard.stat().st_size > DEFAULT_MAX_FILE_BYTES:
            raise ValueError(f"checkpoint shard exceeds size limit: {shard.name}")
        # SafeTensors currently ships no complete typing for safe_open.
        with safe_open(shard, framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
            for key in sorted(handle.keys()):
                if key in hashes:
                    raise ValueError(f"duplicate tensor key across checkpoint shards: {key}")
                hashes[key] = tensor_content_hash(handle.get_tensor(key))
    fingerprint = hash_canonical(
        {
            "schema_version": 1,
            "tensor_hashes": hashes,
        }
    )
    return fingerprint, dict(sorted(hashes.items()))


def fingerprint_files(root: str | Path, filenames: Iterable[str]) -> str:
    directory = Path(root)
    records = []
    for name in sorted(set(filenames)):
        path = directory / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"identity input must be a regular file: {path}")
            records.append(
                {
                    "name": name,
                    "sha256": sha256_file(path, max_bytes=MAX_TOKENIZER_FILE_BYTES),
                }
            )
    return hash_canonical({"files": records, "schema_version": 1})


def tokenizer_fingerprint(checkpoint: str | Path) -> str:
    root = Path(checkpoint)
    if root.is_file():
        return hash_canonical({"files": [], "schema_version": 1})
    return fingerprint_files(root, TOKENIZER_FILENAMES)


def chat_template_fingerprint(checkpoint: str | Path) -> str:
    root = Path(checkpoint)
    path = root / "tokenizer_config.json" if root.is_dir() else Path()
    template: object = None
    if path.is_file():
        value = _bounded_json(path)
        if isinstance(value, dict):
            template = value.get("chat_template")
    return hash_canonical({"chat_template": template})


def generation_config_fingerprint(checkpoint: str | Path) -> str:
    root = Path(checkpoint)
    path = root / "generation_config.json" if root.is_dir() else Path()
    value: object = {}
    if path.is_file():
        value = _bounded_json(path)
    return hash_canonical(value)
