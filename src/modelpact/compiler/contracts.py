"""Prepare data-only Behavior Contracts for trusted adapter optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from modelpact.adapters.base import ModelAdapter, ModelBatch
from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
from modelpact.contracts.ast import AssertionType, BehaviorContract, CompileObjective, ObjectiveType
from modelpact.contracts.objectives import ObjectiveInputs, evaluate_objective
from modelpact.contracts.parser import resolve_contract_resource
from modelpact.util.canonical_json import CanonicalJSONError, strict_json_loads
from modelpact.util.hashing import sha256_file

MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 1_000_000


class ContractPreparationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CompilationRecord:
    record_id: str
    prompt: str
    target: str | None
    preferred: str | None
    dispreferred: str | None
    teacher_logits: Tensor | None
    reference_logits: Tensor | None
    reference_hidden_states: Tensor | None
    direction: Tensor | None


@dataclass(frozen=True, slots=True)
class PreparedExample:
    record: CompilationRecord
    batch: ModelBatch
    labels: Tensor | None = None
    token_mask: Tensor | None = None
    base_logits: Tensor | None = None


@dataclass(frozen=True, slots=True)
class PreparedContract:
    contract: BehaviorContract
    objectives: tuple[DifferentiableObjective, ...]
    guards: tuple[DifferentiableConstraint, ...]
    source_hashes: dict[str, str]
    record_counts: dict[str, int]


_ALLOWED_RECORD_FIELDS = {
    "id",
    "prompt",
    "target",
    "reference",
    "preferred",
    "dispreferred",
    "teacher_logits",
    "reference_logits",
    "reference_hidden_states",
    "direction",
    # Verification-only values can share the same source and are ignored here.
    "expected",
    "choices",
    "correct_choice",
    "values",
}


def _optional_string(record: dict[str, object], name: str, *, line: int) -> str | None:
    value = record.get(name)
    if value is not None and not isinstance(value, str):
        raise ContractPreparationError(f"line {line}: {name} must be a string")
    return value


def _optional_tensor(record: dict[str, object], name: str, *, line: int) -> Tensor | None:
    value = record.get(name)
    if value is None:
        return None
    try:
        tensor = torch.tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ContractPreparationError(
            f"line {line}: {name} must be a finite numeric array"
        ) from error
    if tensor.numel() > 100_000_000 or not bool(torch.isfinite(tensor).all()):
        raise ContractPreparationError(f"line {line}: {name} exceeds limits or is non-finite")
    return tensor


def load_compilation_records(
    contract_path: str | Path, source: str
) -> tuple[CompilationRecord, ...]:
    path = resolve_contract_resource(contract_path, source)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ContractPreparationError(f"invalid or oversized objective source: {source}")
    records: list[CompilationRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                raise ContractPreparationError(f"line {line_number}: record exceeds size limit")
            try:
                value = strict_json_loads(line)
            except CanonicalJSONError as error:
                raise ContractPreparationError(f"line {line_number}: malformed JSON") from error
            if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
                raise ContractPreparationError(f"line {line_number}: record must be an object")
            unknown = set(value) - _ALLOWED_RECORD_FIELDS
            if unknown:
                raise ContractPreparationError(
                    f"line {line_number}: unknown fields {sorted(unknown)}"
                )
            prompt = value.get("prompt")
            if not isinstance(prompt, str) or not prompt or len(prompt) > 1_000_000:
                raise ContractPreparationError(
                    f"line {line_number}: prompt must be a bounded nonempty string"
                )
            record_id = value.get("id", f"record-{line_number:06d}")
            if not isinstance(record_id, str) or not record_id:
                raise ContractPreparationError(f"line {line_number}: id must be a nonempty string")
            target = _optional_string(value, "target", line=line_number)
            if target is None:
                target = _optional_string(value, "reference", line=line_number)
            records.append(
                CompilationRecord(
                    record_id,
                    prompt,
                    target,
                    _optional_string(value, "preferred", line=line_number),
                    _optional_string(value, "dispreferred", line=line_number),
                    _optional_tensor(value, "teacher_logits", line=line_number),
                    _optional_tensor(value, "reference_logits", line=line_number),
                    _optional_tensor(value, "reference_hidden_states", line=line_number),
                    _optional_tensor(value, "direction", line=line_number),
                )
            )
            if len(records) > MAX_RECORDS:
                raise ContractPreparationError(f"source exceeds {MAX_RECORDS} records")
    if not records:
        raise ContractPreparationError(f"objective source is empty: {source}")
    return tuple(records)


def _target_example(adapter: ModelAdapter, record: CompilationRecord) -> PreparedExample:
    if record.target is None:
        raise ContractPreparationError(
            f"record {record.record_id}: teacher_cross_entropy requires target/reference"
        )
    tokenizer = adapter.tokenizer()
    prompt_ids = tokenizer.encode(record.prompt, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(record.target, add_bos=False, add_eos=True)
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    mask = torch.zeros_like(attention)
    mask[:, len(prompt_ids) :] = True
    return PreparedExample(record, ModelBatch(input_ids, attention), labels=labels, token_mask=mask)


def _prompt_example(
    adapter: ModelAdapter,
    base_model: nn.Module,
    record: CompilationRecord,
    *,
    capture_base: bool,
) -> PreparedExample:
    batch = adapter.tokenizer().batch([record.prompt])
    base_logits = None
    if capture_base:
        with torch.no_grad():
            base_logits = adapter.forward_logits(base_model, batch).detach().cpu()
    return PreparedExample(record, batch, token_mask=batch.attention_mask, base_logits=base_logits)


def _sequence_log_probability(
    adapter: ModelAdapter,
    model: nn.Module,
    prompt: str,
    completion: str,
) -> Tensor:
    tokenizer = adapter.tokenizer()
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    completion_ids = tokenizer.encode(completion, add_bos=False, add_eos=True)
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    batch = ModelBatch(ids, torch.ones_like(ids, dtype=torch.bool))
    logits = adapter.forward_logits(model, batch)
    log_probabilities = torch.log_softmax(logits[:, :-1, :].to(torch.float64), dim=-1)
    labels = ids[:, 1:].to(log_probabilities.device)
    selected = log_probabilities.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(0, len(prompt_ids) - 1)
    return selected[:, start:].sum(dim=-1)


def _activation(
    adapter: ModelAdapter,
    model: nn.Module,
    batch: ModelBatch,
    activation_point: str,
) -> Tensor:
    points = {point.path: point.module for point in adapter.activation_points(model)}
    module = points.get(activation_point)
    if module is None:
        raise ContractPreparationError(
            f"adapter does not expose activation point {activation_point!r}"
        )
    captured: list[Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        candidate = output[0] if isinstance(output, tuple) else output
        if isinstance(candidate, Tensor):
            captured.append(candidate)

    handle = module.register_forward_hook(hook)
    try:
        adapter.forward_logits(model, batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ContractPreparationError(
            f"activation point {activation_point!r} did not produce one tensor"
        )
    return captured[0]


def _objective_loss(
    adapter: ModelAdapter,
    specification: CompileObjective,
) -> Callable[[nn.Module, PreparedExample], Tensor]:
    def loss(model: nn.Module, example: PreparedExample) -> Tensor:
        logits: Tensor | None = None
        preferred: Tensor | None = None
        dispreferred: Tensor | None = None
        hidden: Tensor | None = None
        activations: Tensor | None = None
        if specification.type is ObjectiveType.PREFERRED_SEQUENCE_MARGIN:
            if example.record.preferred is None or example.record.dispreferred is None:
                raise ContractPreparationError(
                    f"record {example.record.record_id}: preferred_sequence_margin "
                    "requires preferred/dispreferred"
                )
            preferred = _sequence_log_probability(
                adapter, model, example.record.prompt, example.record.preferred
            )
            dispreferred = _sequence_log_probability(
                adapter, model, example.record.prompt, example.record.dispreferred
            )
        elif specification.type in {
            ObjectiveType.HIDDEN_STATE_MATCHING,
            ObjectiveType.ACTIVATION_DIRECTION,
        }:
            point = specification.options.get("activation_point")
            if not isinstance(point, str):
                raise ContractPreparationError(
                    f"objective {specification.id}: activation_point is required"
                )
            captured = _activation(adapter, model, example.batch, point)
            if specification.type is ObjectiveType.HIDDEN_STATE_MATCHING:
                hidden = captured
            else:
                activations = captured
        else:
            logits = adapter.forward_logits(model, example.batch)
        teacher_logits = example.record.teacher_logits
        if teacher_logits is not None and logits is not None:
            teacher_logits = teacher_logits.to(device=logits.device, dtype=logits.dtype)
            if teacher_logits.ndim == 2:
                teacher_logits = teacher_logits.unsqueeze(0)
        evaluation = evaluate_objective(
            specification,
            ObjectiveInputs(
                logits=logits,
                labels=None
                if example.labels is None
                else example.labels.to(
                    logits.device if logits is not None else example.labels.device
                ),
                teacher_logits=teacher_logits,
                base_logits=None
                if example.base_logits is None or logits is None
                else example.base_logits.to(device=logits.device, dtype=logits.dtype),
                token_mask=None
                if example.token_mask is None or logits is None
                else example.token_mask.to(logits.device),
                preferred_log_prob=preferred,
                dispreferred_log_prob=dispreferred,
                hidden_states=hidden,
                reference_hidden_states=None
                if example.record.reference_hidden_states is None or hidden is None
                else example.record.reference_hidden_states.to(
                    device=hidden.device, dtype=hidden.dtype
                ),
                activations=activations,
                direction=None
                if example.record.direction is None or activations is None
                else example.record.direction.to(
                    device=activations.device, dtype=activations.dtype
                ),
            ),
        )
        if not evaluation.supported or evaluation.loss is None:
            raise ContractPreparationError(
                f"objective {specification.id} could not execute: "
                f"{evaluation.reason or evaluation.outcome.value}"
            )
        return evaluation.loss

    return loss


def _kl_measure(
    adapter: ModelAdapter,
    reference_name: str,
) -> Callable[[nn.Module, PreparedExample], Tensor]:
    def measure(model: nn.Module, example: PreparedExample) -> Tensor:
        student = adapter.forward_logits(model, example.batch).to(torch.float64)
        if reference_name == "base_logits":
            reference = example.base_logits
        else:
            reference = example.record.reference_logits
        if reference is None:
            raise ContractPreparationError(
                f"record {example.record.record_id}: missing {reference_name}"
            )
        reference = reference.to(device=student.device, dtype=student.dtype)
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if reference.shape != student.shape:
            raise ContractPreparationError(
                f"record {example.record.record_id}: reference/student logit shapes differ"
            )
        teacher_log = torch.log_softmax(reference, dim=-1)
        student_log = torch.log_softmax(student, dim=-1)
        per_token = (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1)
        mask = example.batch.attention_mask.to(device=per_token.device, dtype=per_token.dtype)
        return (per_token * mask).sum() / mask.sum().clamp_min(1)

    return measure


def _maximum_from_assertion(options: dict[str, object]) -> float:
    candidates: list[float] = []
    for key in ("maximum_mean", "maximum_item", "maximum"):
        value = options.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            candidates.append(float(value))
    quantile = options.get("maximum_quantile")
    if isinstance(quantile, dict) and isinstance(quantile.get("value"), int | float):
        candidates.append(float(cast(int | float, quantile["value"])))
    if not candidates:
        raise ContractPreparationError("KL preservation assertion has no maximum threshold")
    return min(candidates)


def prepare_contract(
    adapter: ModelAdapter,
    base_model: nn.Module,
    contract: BehaviorContract,
    contract_path: str | Path,
) -> PreparedContract:
    cache: dict[str, tuple[CompilationRecord, ...]] = {}
    hashes: dict[str, str] = {}

    def records(source: str) -> tuple[CompilationRecord, ...]:
        if source not in cache:
            cache[source] = load_compilation_records(contract_path, source)
            hashes[source] = sha256_file(resolve_contract_resource(contract_path, source))
        return cache[source]

    objectives: list[DifferentiableObjective] = []
    for specification in contract.objectives:
        source_records = records(specification.source)
        if specification.type is ObjectiveType.TEACHER_CROSS_ENTROPY:
            examples = tuple(_target_example(adapter, record) for record in source_records)
        elif specification.type is ObjectiveType.BASE_KL:
            examples = tuple(
                _prompt_example(adapter, base_model, record, capture_base=True)
                for record in source_records
            )
        else:
            examples = tuple(
                _prompt_example(adapter, base_model, record, capture_base=False)
                for record in source_records
            )
        objectives.append(
            DifferentiableObjective(
                specification.id,
                examples,
                _objective_loss(adapter, specification),
                weight=specification.weight,
            )
        )
    guards: list[DifferentiableConstraint] = []
    for assertion in contract.guards:
        if assertion.type not in {AssertionType.BASE_KL, AssertionType.REFERENCE_KL}:
            continue
        source_records = records(assertion.source)
        examples = tuple(
            _prompt_example(
                adapter,
                base_model,
                record,
                capture_base=assertion.type is AssertionType.BASE_KL,
            )
            for record in source_records
        )
        guards.append(
            DifferentiableConstraint(
                assertion.id,
                examples,
                _kl_measure(
                    adapter,
                    "base_logits"
                    if assertion.type is AssertionType.BASE_KL
                    else "reference_logits",
                ),
                maximum=_maximum_from_assertion(dict(assertion.options)),
            )
        )
    return PreparedContract(
        contract,
        tuple(objectives),
        tuple(guards),
        dict(sorted(hashes.items())),
        {source: len(values) for source, values in sorted(cache.items())},
    )
