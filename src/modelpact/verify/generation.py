"""Deterministic free-generation execution and evidence records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from modelpact.contracts.assertions import EvaluationRecord, MetricValue
from modelpact.contracts.ast import GenerationPolicy
from modelpact.util.hashing import hash_canonical, sha256_bytes


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    sample_id: str
    prompt: str

    def __post_init__(self) -> None:
        if not self.sample_id or len(self.sample_id) > 4096:
            raise ValueError("sample_id must be a non-empty bounded string")
        if len(self.prompt) > 1_000_000 or "\x00" in self.prompt:
            raise ValueError("prompt exceeds generation limits")


@dataclass(frozen=True, slots=True)
class GeneratedOutput:
    text: str
    token_ids: tuple[int, ...]
    token_log_probabilities: tuple[float, ...] = ()
    parser_result: Mapping[str, MetricValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.text) > 1_000_000:
            raise ValueError("generated output exceeds generation limits")
        if len(self.token_ids) > 1_000_000 or any(
            isinstance(token, bool) or token < 0 for token in self.token_ids
        ):
            raise ValueError("generated token IDs exceed generation limits")
        if self.token_log_probabilities and len(self.token_log_probabilities) != len(
            self.token_ids
        ):
            raise ValueError("one log probability is required per generated token")
        if any(not math.isfinite(value) for value in self.token_log_probabilities):
            raise ValueError("token log probabilities must be finite")
        if len(self.parser_result) > 1000:
            raise ValueError("parser result exceeds generation limits")


class GenerationBackend(Protocol):
    """Trusted adapter boundary.  Contract data never chooses an implementation."""

    def generate(
        self,
        prompt: str,
        *,
        policy: GenerationPolicy,
        seed: int,
    ) -> GeneratedOutput: ...


@dataclass(frozen=True, slots=True)
class FreeGenerationRecord:
    sample_id: str
    prompt_hash: str
    output_hash: str
    seed: int
    generation_policy_hash: str
    token_ids_hash: str
    generated_tokens: int
    parser_result: Mapping[str, MetricValue]
    token_diagnostics: tuple[Mapping[str, int | float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "prompt_hash": self.prompt_hash,
            "output_hash": self.output_hash,
            "seed": self.seed,
            "generation_policy_hash": self.generation_policy_hash,
            "token_ids_hash": self.token_ids_hash,
            "generated_tokens": self.generated_tokens,
            "parser_result": dict(sorted(self.parser_result.items())),
            "token_diagnostics": [dict(item) for item in self.token_diagnostics],
        }


@dataclass(frozen=True, slots=True)
class GenerationExecution:
    records: tuple[FreeGenerationRecord, ...]
    evaluation_records: tuple[EvaluationRecord, ...]


def _token_diagnostics(output: GeneratedOutput) -> tuple[Mapping[str, int | float], ...]:
    if not output.token_log_probabilities:
        return tuple(
            {"index": index, "token_id": token_id}
            for index, token_id in enumerate(output.token_ids)
        )
    return tuple(
        {
            "index": index,
            "token_id": token_id,
            "log_probability": log_probability,
        }
        for index, (token_id, log_probability) in enumerate(
            zip(output.token_ids, output.token_log_probabilities, strict=True)
        )
    )


def record_generated_output(
    request: GenerationRequest,
    output: GeneratedOutput,
    *,
    policy: GenerationPolicy,
    seed: int,
) -> FreeGenerationRecord:
    token_ids_hash = hash_canonical(list(output.token_ids))
    return FreeGenerationRecord(
        sample_id=request.sample_id,
        prompt_hash=sha256_bytes(request.prompt.encode("utf-8")),
        output_hash=sha256_bytes(output.text.encode("utf-8")),
        seed=seed,
        generation_policy_hash=hash_canonical(policy.to_dict()),
        token_ids_hash=token_ids_hash,
        generated_tokens=len(output.token_ids),
        parser_result=dict(sorted(output.parser_result.items())),
        token_diagnostics=_token_diagnostics(output),
    )


def execute_free_generation(
    backend: GenerationBackend,
    requests: Sequence[GenerationRequest],
    *,
    policy: GenerationPolicy,
) -> GenerationExecution:
    """Execute every request/seed pair in deterministic request order."""

    identifiers = [request.sample_id for request in requests]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("generation request identifiers must be unique")
    evidence: list[FreeGenerationRecord] = []
    evaluations: list[EvaluationRecord] = []
    for request in sorted(requests, key=lambda item: item.sample_id):
        for seed in policy.seeds:
            output = backend.generate(request.prompt, policy=policy, seed=seed)
            evidence.append(record_generated_output(request, output, policy=policy, seed=seed))
            evaluations.append(
                EvaluationRecord(
                    sample_id=f"{request.sample_id}:seed-{seed}",
                    prompt=request.prompt,
                    generated_text=output.text,
                    generated_token_ids=output.token_ids,
                    values={"generation_seed": seed, **output.parser_result},
                )
            )
    return GenerationExecution(records=tuple(evidence), evaluation_records=tuple(evaluations))


__all__ = [
    "FreeGenerationRecord",
    "GeneratedOutput",
    "GenerationBackend",
    "GenerationExecution",
    "GenerationRequest",
    "execute_free_generation",
    "record_generated_output",
]
