"""Safe loading and role separation for probe datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from modelpact.util.hashing import hash_canonical

MAX_JSONL_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_PROBES = 1_000_000


class ProbeDataError(ValueError):
    pass


class ProbeRole(StrEnum):
    COMPILE = "compile"
    SEARCH = "search"
    VALIDATION = "validation"
    HOLDOUT = "holdout"
    GUARD = "guard"


@dataclass(frozen=True, slots=True)
class Probe:
    probe_id: str
    prompt: str
    role: ProbeRole
    reference: str | None = None
    choices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_hash(self) -> str:
        return hash_canonical({"prompt": self.prompt})

    def to_dict(self, *, include_prompt: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "probe_id": self.probe_id,
            "prompt_hash": self.prompt_hash,
            "role": self.role.value,
            "reference": self.reference,
            "choices": list(self.choices),
            "metadata": self.metadata,
        }
        if include_prompt:
            result["prompt"] = self.prompt
        return result


def _validate_record(record: object, *, line_number: int, default_role: ProbeRole) -> Probe:
    if not isinstance(record, dict):
        raise ProbeDataError(f"line {line_number}: probe must be an object")
    allowed = {"id", "probe_id", "prompt", "role", "reference", "choices", "metadata"}
    unknown = set(record) - allowed
    if unknown:
        raise ProbeDataError(f"line {line_number}: unknown fields: {sorted(unknown)}")
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ProbeDataError(f"line {line_number}: prompt must be a nonempty string")
    if len(prompt.encode("utf-8")) > MAX_LINE_BYTES:
        raise ProbeDataError(f"line {line_number}: prompt is too large")
    identifier = record.get("probe_id", record.get("id", f"probe-{line_number:06d}"))
    if not isinstance(identifier, str) or not identifier:
        raise ProbeDataError(f"line {line_number}: id must be a nonempty string")
    try:
        role = ProbeRole(record.get("role", default_role.value))
    except (TypeError, ValueError) as error:
        raise ProbeDataError(f"line {line_number}: invalid role") from error
    reference = record.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise ProbeDataError(f"line {line_number}: reference must be a string")
    raw_choices = record.get("choices", [])
    if not isinstance(raw_choices, list) or not all(isinstance(item, str) for item in raw_choices):
        raise ProbeDataError(f"line {line_number}: choices must be strings")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ProbeDataError(f"line {line_number}: metadata must be an object")
    return Probe(identifier, prompt, role, reference, tuple(raw_choices), metadata)


def load_jsonl(
    path: str | Path,
    *,
    default_role: ProbeRole = ProbeRole.VALIDATION,
    allow_holdout: bool = False,
    max_probes: int = MAX_PROBES,
) -> tuple[Probe, ...]:
    source = Path(path)
    if source.stat().st_size > MAX_JSONL_BYTES:
        raise ProbeDataError(f"probe file exceeds {MAX_JSONL_BYTES} bytes")
    probes: list[Probe] = []
    identifiers: set[str] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ProbeDataError(f"line {line_number}: record is too large")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProbeDataError(f"line {line_number}: malformed JSON") from error
            probe = _validate_record(record, line_number=line_number, default_role=default_role)
            if probe.role is ProbeRole.HOLDOUT and not allow_holdout:
                raise PermissionError(
                    "sealed holdout probes require an explicit final-candidate gate"
                )
            if probe.probe_id in identifiers:
                raise ProbeDataError(f"line {line_number}: duplicate probe id {probe.probe_id!r}")
            identifiers.add(probe.probe_id)
            probes.append(probe)
            if len(probes) > max_probes:
                raise ProbeDataError(f"probe count exceeds {max_probes}")
    return tuple(probes)


def probes_hash(probes: tuple[Probe, ...]) -> str:
    return hash_canonical([probe.to_dict() for probe in probes])
