"""Typed, immutable representation of Behavior Contract v1.

The AST deliberately separates differentiable compilation objectives from
acceptance assertions.  Instances are safe data: they contain no callables and
their canonical representation is suitable for content addressing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

from modelpact.util.hashing import hash_canonical

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ObjectiveType(StrEnum):
    TEACHER_CROSS_ENTROPY = "teacher_cross_entropy"
    TEACHER_KL = "teacher_kl"
    PREFERRED_SEQUENCE_MARGIN = "preferred_sequence_margin"
    BASE_KL = "base_kl"
    HIDDEN_STATE_MATCHING = "hidden_state_matching"
    ACTIVATION_DIRECTION = "activation_direction"


class AssertionType(StrEnum):
    # Security-scanner false positive: this is a public metric name, not a secret.
    TOKEN_LOG_PROBABILITY = "token_log_probability"  # noqa: S105
    SEQUENCE_LOG_PROBABILITY = "sequence_log_probability"
    SEQUENCE_MARGIN = "sequence_margin"
    MULTIPLE_CHOICE_MARGIN = "multiple_choice_margin"
    EXACT_MATCH = "exact_match"
    NORMALIZED_EXACT_MATCH = "normalized_exact_match"
    REGULAR_EXPRESSION = "regular_expression"
    JSON_PARSE = "json_parse"
    JSON_SCHEMA = "json_schema"
    FREE_GENERATION_MATCH = "free_generation_match"
    REFERENCE_KL = "reference_kl"
    BASE_KL = "base_kl"
    GENERATION_LENGTH = "generation_length"
    PERPLEXITY = "perplexity"


class UnsealPolicy(StrEnum):
    FINAL_CANDIDATE_ONLY = "final_candidate_only"
    INDEPENDENT_VERIFICATION = "independent_verification"


class GenerationMode(StrEnum):
    GREEDY = "greedy"
    SAMPLE = "sample"


def _check_id(value: str, *, field_name: str) -> None:
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_ID_PATTERN.pattern!r} and be at most 128 characters"
        )


def _check_hash(value: str | None, *, field_name: str) -> None:
    if value is not None and not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase sha256: digest")


def _freeze_json(value: JsonValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return cast(JsonScalar, value)


def _frozen_options(options: Mapping[str, JsonValue]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze_json(value) for key, value in options.items()})


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    """Exact identities required for execution when declared."""

    tokenizer_hash: str | None = None
    base_signature: str | None = None
    architecture_hash: str | None = None
    state_schema_hash: str | None = None
    adapter_id: str | None = None
    output_semantics: str = "causal_lm"

    def __post_init__(self) -> None:
        for name in ("tokenizer_hash", "architecture_hash", "state_schema_hash"):
            _check_hash(getattr(self, name), field_name=name)
        if self.base_signature is not None and not self.base_signature:
            raise ValueError("base_signature must not be empty")
        if self.adapter_id is not None and not self.adapter_id:
            raise ValueError("adapter_id must not be empty")
        if self.output_semantics != "causal_lm":
            raise ValueError("Behavior Contract v1 supports only causal_lm output semantics")

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"output_semantics": self.output_semantics}
        for name in (
            "tokenizer_hash",
            "base_signature",
            "architecture_hash",
            "state_schema_hash",
            "adapter_id",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class CompileObjective:
    id: str
    type: ObjectiveType
    source: str
    weight: float = 1.0
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_id(self.id, field_name="objective id")
        if not self.source or len(self.source) > 4096 or "\x00" in self.source:
            raise ValueError("objective source must be a non-empty bounded string")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("objective weight must be finite and positive")
        object.__setattr__(
            self,
            "options",
            _frozen_options(cast(Mapping[str, JsonValue], self.options)),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "weight": self.weight,
        }
        result.update({key: _thaw_json(value) for key, value in self.options.items()})
        return result


@dataclass(frozen=True, slots=True)
class VerificationAssertion:
    id: str
    type: AssertionType
    source: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_id(self.id, field_name="assertion id")
        if not self.source or len(self.source) > 4096 or "\x00" in self.source:
            raise ValueError("assertion source must be a non-empty bounded string")
        object.__setattr__(
            self,
            "options",
            _frozen_options(cast(Mapping[str, JsonValue], self.options)),
        )

    def option(self, name: str, default: object = None) -> object:
        return self.options.get(name, default)

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
        }
        result.update({key: _thaw_json(value) for key, value in self.options.items()})
        return result


@dataclass(frozen=True, slots=True)
class HoldoutPolicy:
    sealed: bool = True
    targets: str | None = None
    guards: str | None = None
    unseal_policy: UnsealPolicy = UnsealPolicy.FINAL_CANDIDATE_ONLY

    def __post_init__(self) -> None:
        if not self.sealed and self.unseal_policy is UnsealPolicy.FINAL_CANDIDATE_ONLY:
            raise ValueError("final_candidate_only requires a sealed holdout")
        for name in ("targets", "guards"):
            value = getattr(self, name)
            if value is not None and (not value or len(value) > 4096 or "\x00" in value):
                raise ValueError(f"holdout {name} source is invalid")

    @property
    def configured(self) -> bool:
        return self.targets is not None or self.guards is not None

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "sealed": self.sealed,
            "unseal_policy": self.unseal_policy.value,
        }
        if self.targets is not None:
            result["targets"] = self.targets
        if self.guards is not None:
            result["guards"] = self.guards
        return result


@dataclass(frozen=True, slots=True)
class StatisticsPolicy:
    confidence_level: float = 0.95
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 81273
    multiple_comparison: str = "none"

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between zero and one")
        if isinstance(self.bootstrap_samples, bool) or not 1 <= self.bootstrap_samples <= 1_000_000:
            raise ValueError("bootstrap_samples must be in [1, 1000000]")
        if isinstance(self.bootstrap_seed, bool) or not 0 <= self.bootstrap_seed < 2**63:
            raise ValueError("bootstrap_seed must be a non-negative signed 64-bit integer")
        if self.multiple_comparison not in {"none", "holm", "bonferroni"}:
            raise ValueError("multiple_comparison must be none, holm, or bonferroni")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "multiple_comparison": self.multiple_comparison,
        }


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    mode: GenerationMode = GenerationMode.GREEDY
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seeds: tuple[int, ...] = (0,)
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or not 1 <= self.max_new_tokens <= 1_000_000:
            raise ValueError("max_new_tokens must be in [1, 1000000]")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not 1 <= self.top_k <= 10_000_000
        ):
            raise ValueError("top_k must be in [1, 10000000]")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not self.seeds or len(self.seeds) > 1024:
            raise ValueError("generation seeds must contain between 1 and 1024 entries")
        if any(isinstance(seed, bool) or not 0 <= seed < 2**63 for seed in self.seeds):
            raise ValueError("generation seeds must be non-negative signed 64-bit integers")
        if len(self.stop_sequences) > 128 or any(
            not value or len(value) > 4096 or "\x00" in value for value in self.stop_sequences
        ):
            raise ValueError("stop sequences exceed Contract v1 limits")
        if self.mode is GenerationMode.GREEDY and (
            self.top_k is not None or self.top_p != 1.0 or self.temperature != 1.0
        ):
            raise ValueError("greedy generation cannot declare sampling parameters")

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "mode": self.mode.value,
            "max_new_tokens": self.max_new_tokens,
            "seeds": list(self.seeds),
        }
        if self.mode is GenerationMode.SAMPLE:
            result["temperature"] = self.temperature
            result["top_p"] = self.top_p
            if self.top_k is not None:
                result["top_k"] = self.top_k
        if self.stop_sequences:
            result["stop_sequences"] = list(self.stop_sequences)
        return result


@dataclass(frozen=True, slots=True)
class BehaviorContract:
    schema_version: int
    id: str
    contract_version: int
    model_requirements: ModelRequirements
    objectives: tuple[CompileObjective, ...]
    targets: tuple[VerificationAssertion, ...]
    guards: tuple[VerificationAssertion, ...]
    holdout: HoldoutPolicy
    statistics: StatisticsPolicy
    generation: GenerationPolicy
    description: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only Behavior Contract schema_version 1 is supported")
        _check_id(self.id, field_name="contract id")
        if isinstance(self.contract_version, bool) or not 1 <= self.contract_version <= 2**31 - 1:
            raise ValueError("contract_version must be a positive signed 32-bit integer")
        if self.description is not None and (
            len(self.description) > 16_384 or "\x00" in self.description
        ):
            raise ValueError("description exceeds Contract v1 limits")
        objective_ids = [item.id for item in self.objectives]
        assertion_ids = [item.id for item in (*self.targets, *self.guards)]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("objective identifiers must be unique")
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("target and guard assertion identifiers must be globally unique")
        holdout_sources = {item for item in (self.holdout.targets, self.holdout.guards) if item}
        visible_sources = {item.source for item in self.objectives}
        visible_sources.update(item.source for item in self.targets)
        visible_sources.update(item.source for item in self.guards)
        overlap = holdout_sources & visible_sources
        if overlap:
            raise ValueError(
                "sealed holdout sources must be distinct from compile/search/validation sources: "
                + ", ".join(sorted(overlap))
            )

    @property
    def contract_id(self) -> str:
        """Content address of the complete normalized contract."""

        return hash_canonical(self.to_dict())

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "contract_version": self.contract_version,
            "model_requirements": self.model_requirements.to_dict(),
            "compile": {"objectives": [item.to_dict() for item in self.objectives]},
            "verify": {
                "targets": [item.to_dict() for item in self.targets],
                "guards": [item.to_dict() for item in self.guards],
            },
            "holdout": self.holdout.to_dict(),
            "statistics": self.statistics.to_dict(),
            "generation": self.generation.to_dict(),
        }
        if self.description is not None:
            result["description"] = self.description
        return result


__all__ = [
    "AssertionType",
    "BehaviorContract",
    "CompileObjective",
    "GenerationMode",
    "GenerationPolicy",
    "HoldoutPolicy",
    "JsonScalar",
    "JsonValue",
    "ModelRequirements",
    "ObjectiveType",
    "StatisticsPolicy",
    "UnsealPolicy",
    "VerificationAssertion",
]
