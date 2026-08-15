"""Difference witness records make no claims beyond executed scopes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from modelpact.util.hashing import hash_canonical


@dataclass(frozen=True, slots=True)
class DifferenceWitness:
    witness_id: str
    input_hash: str
    original_input: str
    minimized_input: str
    divergence_metrics: dict[str, float]
    base_output_hash: str
    target_output_hash: str
    activation_fingerprint: tuple[float, ...] = ()
    gradient_fingerprint: tuple[float, ...] = ()
    prompt_fingerprint: tuple[float, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        original_input: str,
        minimized_input: str,
        divergence_metrics: dict[str, float],
        base_output: object,
        target_output: object,
        activation_fingerprint: tuple[float, ...] = (),
        gradient_fingerprint: tuple[float, ...] = (),
        prompt_fingerprint: tuple[float, ...] = (),
        provenance: dict[str, Any] | None = None,
    ) -> DifferenceWitness:
        input_hash = hash_canonical({"input": minimized_input})
        payload = {
            "input_hash": input_hash,
            "divergence_metrics": divergence_metrics,
            "base_output_hash": hash_canonical(base_output),
            "target_output_hash": hash_canonical(target_output),
            "provenance": provenance or {},
        }
        return cls(
            witness_id=hash_canonical(payload),
            input_hash=input_hash,
            original_input=original_input,
            minimized_input=minimized_input,
            divergence_metrics=divergence_metrics,
            base_output_hash=payload["base_output_hash"],
            target_output_hash=payload["target_output_hash"],
            activation_fingerprint=activation_fingerprint,
            gradient_fingerprint=gradient_fingerprint,
            prompt_fingerprint=prompt_fingerprint,
            provenance=provenance or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def witness_set_hash(witnesses: tuple[DifferenceWitness, ...]) -> str:
    return hash_canonical([witness.to_dict() for witness in sorted(witnesses, key=lambda item: item.witness_id)])

