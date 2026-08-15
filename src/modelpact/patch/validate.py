"""Untrusted patch data parsing and resource limits."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from modelpact.models.manifest import ModelSignature
from modelpact.patch.ast import DeltaProgram
from modelpact.util.canonical_json import CanonicalJSONError, strict_json_loads

MAX_DELTA_PROGRAM_BYTES = 16 * 1024**2
BASE_IDENTITY_FIELDS = (
    "adapter_id",
    "architecture_hash",
    "state_schema_hash",
    "checkpoint_hash",
    "tokenizer_hash",
    "chat_template_hash",
    "generation_config_hash",
)


def load_delta_program(path: str | Path) -> DeltaProgram:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"delta program must be a regular file: {source}")
    if source.stat().st_size > MAX_DELTA_PROGRAM_BYTES:
        raise ValueError("delta program exceeds size limit")
    try:
        value = strict_json_loads(source.read_bytes())
    except (CanonicalJSONError, RecursionError) as error:
        raise ValueError("malformed delta program JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("delta program must be a JSON object")
    return DeltaProgram.from_dict(value)


def validate_base_signature(
    expected: Mapping[str, object],
    actual: ModelSignature | Mapping[str, object],
) -> None:
    """Require exact apply-time model identity for all compatibility dimensions."""

    actual_value = actual.to_dict() if isinstance(actual, ModelSignature) else actual
    missing = [field for field in BASE_IDENTITY_FIELDS if field not in expected]
    if missing:
        raise ValueError(f"patch base signature is incomplete: {missing}")
    mismatches = {
        field: {"actual": actual_value.get(field), "expected": expected.get(field)}
        for field in BASE_IDENTITY_FIELDS
        if expected.get(field) != actual_value.get(field)
    }
    if mismatches:
        raise ValueError(f"patch base signature mismatch: {mismatches}")
