"""Bounded local probe loading and real model-backed assertion records."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn

from modelpact.adapters.base import (
    GenerationPolicy as AdapterGenerationPolicy,
)
from modelpact.adapters.base import ModelAdapter, ModelBatch
from modelpact.contracts.assertions import EvaluationRecord
from modelpact.contracts.ast import (
    AssertionType,
    GenerationMode,
    GenerationPolicy,
    VerificationAssertion,
)
from modelpact.contracts.holdout import HoldoutCapability
from modelpact.contracts.parser import ContractLimits, loads_data
from modelpact.util.hashing import sha256_bytes, sha256_file
from modelpact.util.paths import resolve_inside, safe_relative_path
from modelpact.verify.engine import UnsupportedRecordProviderError, VerificationRole
from modelpact.verify.generation import (
    FreeGenerationRecord,
    GeneratedOutput,
    GenerationRequest,
    record_generated_output,
)


@dataclass(frozen=True, slots=True)
class ProbeLimits:
    max_file_bytes: int = 64 * 1024 * 1024
    max_line_bytes: int = 2 * 1024 * 1024
    max_records: int = 100_000
    max_prompt_characters: int = 1_000_000
    max_completion_characters: int = 1_000_000
    max_choices: int = 10_000
    max_logit_values: int = 10_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_PROBE_LIMITS = ProbeLimits()


_LINE_LIMITS = ContractLimits(
    max_bytes=2 * 1024 * 1024,
    max_depth=32,
    max_nodes=100_000,
    max_string_length=1_000_000,
    max_object_keys=10_000,
    max_objectives=1,
    max_assertions=1,
)
_ALLOWED_PROBE_FIELDS = {
    "id",
    "prompt",
    "completion",
    "expected",
    "pattern",
    "choices",
    "correct_choice",
    "preferred",
    "dispreferred",
    "token_id",
    "position",
    "input_hash",
    "reference_logits",
}
_GENERATIVE_TYPES = frozenset(
    {
        AssertionType.EXACT_MATCH,
        AssertionType.NORMALIZED_EXACT_MATCH,
        AssertionType.REGULAR_EXPRESSION,
        AssertionType.JSON_PARSE,
        AssertionType.JSON_SCHEMA,
        AssertionType.FREE_GENERATION_MATCH,
        AssertionType.GENERATION_LENGTH,
    }
)


class ProbeDataError(ValueError):
    pass


def _regular_file_inside(root: Path, relative: str) -> Path:
    safe_relative_path(relative)
    path = resolve_inside(root, relative)
    resolved_root = root.resolve()
    current = path
    while current != resolved_root:
        if current.is_symlink():
            raise ProbeDataError(f"probe resources cannot be symbolic links: {relative}")
        if current.parent == current:
            raise ProbeDataError(f"probe path is outside its declared root: {relative}")
        current = current.parent
    if not path.is_file():
        raise ProbeDataError(f"probe source is not a regular file: {relative}")
    return path


def _probe_mapping(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ProbeDataError(f"{location}: probe record must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ProbeDataError(f"{location}: probe keys must be strings")
    result = dict(cast(Mapping[str, object], value))
    unknown = set(result) - _ALLOWED_PROBE_FIELDS
    if unknown:
        raise ProbeDataError(f"{location}: unknown probe field(s): {sorted(unknown)}")
    prompt = result.get("prompt")
    if not isinstance(prompt, str):
        raise ProbeDataError(f"{location}: prompt must be a string")
    identifier = result.get("id")
    if identifier is not None and (not isinstance(identifier, str) or not identifier):
        raise ProbeDataError(f"{location}: id must be a non-empty string")
    return result


def load_probe_records(
    root: str | Path,
    source: str,
    *,
    limits: ProbeLimits = DEFAULT_PROBE_LIMITS,
) -> tuple[dict[str, object], ...]:
    """Load strict JSONL or a JSON array without following symlinks."""

    base = Path(root).resolve()
    path = _regular_file_inside(base, source)
    size = path.stat().st_size
    if size > limits.max_file_bytes:
        raise ProbeDataError(f"probe source exceeds {limits.max_file_bytes} bytes")
    suffix = path.suffix.lower()
    records: list[dict[str, object]] = []
    if suffix == ".jsonl":
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if len(line) > limits.max_line_bytes:
                    raise ProbeDataError(f"{source}:{line_number}: line exceeds size limit")
                if not line.strip():
                    continue
                value = loads_data(line, format="json", limits=_LINE_LIMITS)
                records.append(_probe_mapping(value, location=f"{source}:{line_number}"))
                if len(records) > limits.max_records:
                    raise ProbeDataError(f"probe source exceeds {limits.max_records} records")
    elif suffix == ".json":
        document_limits = ContractLimits(
            max_bytes=limits.max_file_bytes,
            max_depth=32,
            max_nodes=max(limits.max_records * 20, 1000),
            max_string_length=max(limits.max_prompt_characters, limits.max_completion_characters),
            max_object_keys=10_000,
            max_objectives=1,
            max_assertions=max(limits.max_records, 1),
        )
        value = loads_data(path.read_bytes(), format="json", limits=document_limits)
        if not isinstance(value, list) or len(value) > limits.max_records:
            raise ProbeDataError("JSON probe source must be a bounded array")
        records.extend(
            _probe_mapping(item, location=f"{source}[{index}]") for index, item in enumerate(value)
        )
    else:
        raise ProbeDataError("probe source must end in .jsonl or .json")
    identifiers: set[str] = set()
    for index, record in enumerate(records):
        prompt = cast(str, record["prompt"])
        if len(prompt) > limits.max_prompt_characters or "\x00" in prompt:
            raise ProbeDataError(f"{source}[{index}]: prompt exceeds limits")
        for name in ("completion", "expected", "preferred", "dispreferred"):
            item = record.get(name)
            if item is not None and (
                not isinstance(item, str)
                or len(item) > limits.max_completion_characters
                or "\x00" in item
            ):
                raise ProbeDataError(f"{source}[{index}].{name} is invalid")
        choices = record.get("choices")
        if choices is not None and (
            not isinstance(choices, list)
            or len(choices) > limits.max_choices
            or not all(isinstance(item, str) and item for item in choices)
        ):
            raise ProbeDataError(f"{source}[{index}].choices is invalid")
        expected_hash = record.get("input_hash")
        observed_hash = sha256_bytes(prompt.encode("utf-8"))
        if expected_hash is not None and expected_hash != observed_hash:
            raise ProbeDataError(f"{source}[{index}]: input_hash does not match prompt")
        identifier = cast(str, record.get("id", f"line-{index:08d}"))
        if identifier in identifiers:
            raise ProbeDataError(f"duplicate probe id {identifier!r}")
        identifiers.add(identifier)
        record["id"] = identifier
    return tuple(records)


def load_json_schemas(
    root: str | Path,
    paths: Sequence[str],
    *,
    max_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    limits = ContractLimits(
        max_bytes=max_bytes,
        max_depth=64,
        max_nodes=200_000,
        max_string_length=1_000_000,
        max_object_keys=50_000,
        max_objectives=1,
        max_assertions=1,
    )
    for relative in sorted(set(paths)):
        path = _regular_file_inside(Path(root).resolve(), relative)
        if path.suffix.lower() != ".json":
            raise ProbeDataError("JSON schemas must use .json files")
        if path.stat().st_size > max_bytes:
            raise ProbeDataError(f"schema exceeds {max_bytes} bytes: {relative}")
        value = loads_data(path.read_bytes(), format="json", limits=limits)
        if not isinstance(value, Mapping):
            raise ProbeDataError(f"JSON schema must be an object: {relative}")
        result[relative] = cast(Mapping[str, object], value)
    return result


class ModelBackedRecordProvider:
    """Execute assertions against patched, base, and reference local models."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        model: nn.Module,
        contract_root: str | Path,
        generation_policy: GenerationPolicy,
        base_model: nn.Module | None = None,
        reference_model: nn.Module | None = None,
        limits: ProbeLimits = DEFAULT_PROBE_LIMITS,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._base_model = base_model
        self._reference_model = reference_model
        self._root = Path(contract_root).resolve()
        self._generation_policy = generation_policy
        self._limits = limits
        self._raw_cache: dict[str, tuple[dict[str, object], ...]] = {}
        self._probe_hashes: dict[str, str] = {}
        self._generation_cache: dict[tuple[str, str, int], EvaluationRecord] = {}
        self._generation_evidence: dict[tuple[str, str, int], FreeGenerationRecord] = {}
        self._lock = threading.RLock()
        adapter.prepare(model)
        if base_model is not None:
            adapter.prepare(base_model)
        if reference_model is not None and reference_model is not base_model:
            adapter.prepare(reference_model)

    @property
    def probe_hashes(self) -> Mapping[str, str]:
        with self._lock:
            return dict(sorted(self._probe_hashes.items()))

    def generation_evidence(self) -> Sequence[FreeGenerationRecord]:
        with self._lock:
            ordered = sorted(self._generation_evidence)
            return tuple(self._generation_evidence[key] for key in ordered)

    def _raw_records(self, source: str) -> tuple[dict[str, object], ...]:
        with self._lock:
            if source not in self._raw_cache:
                path = _regular_file_inside(self._root, source)
                self._raw_cache[source] = load_probe_records(
                    self._root, source, limits=self._limits
                )
                self._probe_hashes[source] = sha256_file(
                    path, max_bytes=self._limits.max_file_bytes
                )
            return self._raw_cache[source]

    def _batch(self, text: str) -> ModelBatch:
        return self._adapter.tokenizer().batch((text,), add_bos=True)

    def _logits(self, model: nn.Module, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._batch(text)
        with torch.no_grad():
            logits = self._adapter.forward_logits(model, batch)
        length = int(batch.attention_mask[0].sum().item())
        return logits[0, :length].detach().cpu(), batch.input_ids[0, :length].detach().cpu()

    def _completion_score(self, model: nn.Module, prompt: str, completion: str) -> float:
        tokenizer = self._adapter.tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        completion_ids = tokenizer.encode(completion, add_bos=False, add_eos=False)
        if not completion_ids:
            raise ProbeDataError("scored completion must contain at least one token")
        combined = prompt_ids + completion_ids
        input_ids = torch.tensor([combined], dtype=torch.long)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            logits = self._adapter.forward_logits(model, ModelBatch(input_ids, attention))[0]
        start = len(prompt_ids)
        positions = torch.arange(start - 1, len(combined) - 1, device=logits.device)
        targets = torch.tensor(completion_ids, dtype=torch.long, device=logits.device)
        scores = functional.log_softmax(logits[positions], dim=-1).gather(1, targets[:, None])
        return float(scores.sum().detach().cpu())

    def _adapter_policy(self, seed: int) -> AdapterGenerationPolicy:
        policy = self._generation_policy
        if policy.max_new_tokens > 4096:
            raise UnsupportedRecordProviderError(
                "adapter protocol currently supports at most 4096 generated tokens"
            )
        sampling_controls = policy.top_k is not None or policy.top_p != 1.0
        if sampling_controls and not bool(
            getattr(self._adapter, "supports_sampling_controls", False)
        ):
            raise UnsupportedRecordProviderError(
                "selected adapter cannot represent top-k/top-p generation"
            )
        if policy.stop_sequences:
            raise UnsupportedRecordProviderError(
                "adapter protocol cannot represent stop-sequence generation"
            )
        return AdapterGenerationPolicy(
            mode="greedy" if policy.mode is GenerationMode.GREEDY else "sample",
            max_new_tokens=policy.max_new_tokens,
            seed=seed,
            temperature=policy.temperature,
            top_k=policy.top_k,
            top_p=policy.top_p,
        )

    def _generated(self, source: str, raw: Mapping[str, object], seed: int) -> EvaluationRecord:
        identifier = cast(str, raw["id"])
        key = (source, identifier, seed)
        with self._lock:
            cached = self._generation_cache.get(key)
            if cached is not None:
                return cached
        prompt = cast(str, raw["prompt"])
        policy = self._adapter_policy(seed)
        batch = self._batch(prompt)
        with torch.no_grad():
            samples = self._adapter.generate(self._model, batch, policy)
        if len(samples) != 1:
            raise RuntimeError("adapter returned an unexpected generation batch size")
        sample = samples[0]
        token_log_probabilities: tuple[float, ...] = ()
        if sample.token_ids:
            try:
                tokenizer = self._adapter.tokenizer()
                prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                combined = prompt_ids + list(sample.token_ids)
                ids = torch.tensor([combined], dtype=torch.long)
                mask = torch.ones_like(ids, dtype=torch.bool)
                with torch.no_grad():
                    logits = self._adapter.forward_logits(self._model, ModelBatch(ids, mask))[0]
                positions = torch.arange(
                    len(prompt_ids) - 1, len(combined) - 1, device=logits.device
                )
                targets = torch.tensor(sample.token_ids, dtype=torch.long, device=logits.device)
                token_scores = functional.log_softmax(logits[positions], dim=-1).gather(
                    1, targets[:, None]
                )
                token_log_probabilities = tuple(float(item) for item in token_scores[:, 0].cpu())
            except (RuntimeError, ValueError):
                token_log_probabilities = ()
        output = GeneratedOutput(
            text=sample.text,
            token_ids=sample.token_ids,
            token_log_probabilities=token_log_probabilities,
            parser_result={"finished": sample.finished},
        )
        request = GenerationRequest(sample_id=identifier, prompt=prompt)
        evidence = record_generated_output(
            request,
            output,
            policy=self._generation_policy,
            seed=seed,
        )
        record_values = {
            name: raw[name]
            for name in (
                "expected",
                "pattern",
                "choices",
                "correct_choice",
                "preferred",
                "dispreferred",
                "token_id",
                "position",
            )
            if name in raw
        }
        record = EvaluationRecord(
            sample_id=f"{identifier}:seed-{seed}",
            prompt=prompt,
            generated_text=sample.text,
            generated_token_ids=sample.token_ids,
            values={
                **record_values,
                "generation_seed": seed,
                "finished": sample.finished,
            },
        )
        with self._lock:
            self._generation_cache[key] = record
            self._generation_evidence[key] = evidence
        return record

    def _scored(
        self, assertion: VerificationAssertion, raw: Mapping[str, object]
    ) -> EvaluationRecord:
        identifier = cast(str, raw["id"])
        prompt = cast(str, raw["prompt"])
        completion = raw.get("completion")
        scored_text = prompt + completion if isinstance(completion, str) else prompt
        logits, ids = self._logits(self._model, scored_text)
        reference_logits = None
        base_logits = None
        if assertion.type is AssertionType.REFERENCE_KL:
            supplied = raw.get("reference_logits")
            if self._reference_model is not None:
                reference_logits, _ = self._logits(self._reference_model, scored_text)
            elif supplied is not None:
                try:
                    reference_logits = torch.tensor(supplied, dtype=logits.dtype)
                except (TypeError, ValueError, RuntimeError) as error:
                    raise ProbeDataError("reference_logits must be a numeric matrix") from error
                if (
                    reference_logits.ndim != 2
                    or reference_logits.shape != logits.shape
                    or reference_logits.numel() > self._limits.max_logit_values
                    or not bool(torch.isfinite(reference_logits).all())
                ):
                    raise ProbeDataError(
                        "reference_logits must be finite, bounded, and match executed logits"
                    )
            else:
                raise UnsupportedRecordProviderError(
                    "reference_kl requires a local reference model or bounded probe logits"
                )
        if assertion.type is AssertionType.BASE_KL:
            if self._base_model is None:
                raise UnsupportedRecordProviderError(
                    "base_kl requires a locally loaded unpatched base model"
                )
            base_logits, _ = self._logits(self._base_model, scored_text)
        values: dict[str, object] = {
            name: raw[name]
            for name in (
                "expected",
                "pattern",
                "choices",
                "correct_choice",
                "preferred",
                "dispreferred",
                "token_id",
                "position",
            )
            if name in raw
        }
        if assertion.type is AssertionType.SEQUENCE_MARGIN:
            preferred = raw.get("preferred", assertion.option("preferred"))
            dispreferred = raw.get("dispreferred", assertion.option("dispreferred"))
            if not isinstance(preferred, str) or not isinstance(dispreferred, str):
                raise ProbeDataError("sequence_margin probes require preferred and dispreferred")
            values["sequence_log_probabilities"] = {
                preferred: self._completion_score(self._model, prompt, preferred),
                dispreferred: self._completion_score(self._model, prompt, dispreferred),
            }
        if assertion.type is AssertionType.MULTIPLE_CHOICE_MARGIN:
            choices = raw.get("choices", assertion.option("choices"))
            if not isinstance(choices, list | tuple) or not all(
                isinstance(item, str) for item in choices
            ):
                raise ProbeDataError("multiple_choice_margin probes require choices")
            values["choice_log_probabilities"] = {
                cast(str, choice): self._completion_score(self._model, prompt, cast(str, choice))
                for choice in choices
            }
        return EvaluationRecord(
            sample_id=identifier,
            prompt=prompt,
            logits=logits,
            input_ids=ids,
            reference_logits=reference_logits,
            base_logits=base_logits,
            values=values,
        )

    def records_for(
        self,
        assertion: VerificationAssertion,
        *,
        source: str,
        role: VerificationRole,
        holdout_capability: HoldoutCapability | None,
    ) -> Sequence[EvaluationRecord]:
        if role in {VerificationRole.HOLDOUT_TARGET, VerificationRole.HOLDOUT_GUARD} and (
            holdout_capability is None
        ):
            raise PermissionError("holdout records require a capability")
        raw_records = self._raw_records(source)
        if assertion.type in _GENERATIVE_TYPES:
            return tuple(
                self._generated(source, raw, seed)
                for raw in raw_records
                for seed in self._generation_policy.seeds
            )
        return tuple(self._scored(assertion, raw) for raw in raw_records)


__all__ = [
    "ModelBackedRecordProvider",
    "ProbeDataError",
    "ProbeLimits",
    "load_json_schemas",
    "load_probe_records",
]
