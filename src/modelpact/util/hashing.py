"""Streaming SHA-256 helpers with explicit algorithm prefixes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from modelpact.util.canonical_json import canonical_json_bytes

CHUNK_SIZE = 1024 * 1024
SHA256_TAGGED_LENGTH = 71


def is_sha256_digest(value: object) -> bool:
    """Return whether *value* is exactly ``sha256:`` plus 64 lowercase hex digits."""

    return (
        isinstance(value, str)
        and len(value) == SHA256_TAGGED_LENGTH
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _tag(digest: str) -> str:
    return f"sha256:{digest}"


def sha256_bytes(data: bytes) -> str:
    return _tag(hashlib.sha256(data).hexdigest())


def sha256_file(path: str | Path, *, max_bytes: int | None = None) -> str:
    source = Path(path)
    size = source.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise ValueError(f"file exceeds maximum size of {max_bytes} bytes: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return _tag(digest.hexdigest())


def hash_canonical(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def hash_parts(parts: Iterable[bytes]) -> str:
    """Hash length-delimited parts so concatenation boundaries are unambiguous."""

    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return _tag(digest.hexdigest())
