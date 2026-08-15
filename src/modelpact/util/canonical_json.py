"""RFC 8785-inspired canonical JSON for content addressing.

ModelPact schemas intentionally exclude non-finite floats and binary values.
Python's JSON encoder then supplies a deterministic UTF-8 representation with
sorted object keys and no insignificant whitespace. This is the normative v1
encoding used by this repository; it is not advertised as a complete JCS
implementation for arbitrary JSON numbers.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


class CanonicalJSONError(ValueError):
    """Raised when a value is outside ModelPact's canonical JSON domain."""


def _normalize(value: Any, *, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        raise CanonicalJSONError(f"maximum nesting depth {max_depth} exceeded")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        return _normalize(value.value, depth=depth, max_depth=max_depth)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError("non-finite numbers are not permitted")
        # Collapse negative zero so semantically equal values hash equally.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError("object keys must be strings")
            normalized[key] = _normalize(item, depth=depth + 1, max_depth=max_depth)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item, depth=depth + 1, max_depth=max_depth) for item in value]
    raise CanonicalJSONError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_dumps(value: Any, *, max_depth: int = 64) -> str:
    """Return the normative ModelPact v1 JSON serialization."""

    normalized = _normalize(value, depth=0, max_depth=max_depth)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any, *, max_depth: int = 64) -> bytes:
    """Return canonical JSON encoded as UTF-8 without a byte-order mark."""

    return canonical_dumps(value, max_depth=max_depth).encode("utf-8")

