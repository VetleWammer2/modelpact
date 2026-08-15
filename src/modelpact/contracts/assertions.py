"""Executable acceptance assertions for causal language-model behavior."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

import torch
import torch.nn.functional as functional

from modelpact.contracts.ast import AssertionType, StatisticsPolicy, VerificationAssertion
from modelpact.contracts.parser import ContractLimits, ContractSyntaxError, loads_data
from modelpact.contracts.statistics import ConfidenceInterval, bootstrap_mean_interval
from modelpact.status import VerificationOutcome
from modelpact.util.hashing import sha256_bytes

MetricValue: TypeAlias = str | int | float | bool | None
BinaryResult: TypeAlias = tuple[
    bool,
    MetricValue,
    float | None,
    Mapping[str, MetricValue],
]
NormalizationForm: TypeAlias = Literal["NFC", "NFD", "NFKC", "NFKD"]

_MAX_GENERATED_TEXT = 1_000_000
_JSON_OUTPUT_LIMITS = ContractLimits(
    max_bytes=2 * 1024 * 1024,
    max_depth=64,
    max_nodes=200_000,
    max_string_length=1_000_000,
    max_object_keys=50_000,
    max_objectives=1,
    max_assertions=1,
)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    sample_id: str
    prompt: str
    generated_text: str | None = None
    generated_token_ids: tuple[int, ...] = ()
    logits: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    reference_logits: torch.Tensor | None = None
    base_logits: torch.Tensor | None = None
    values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or len(self.sample_id) > 4096 or "\x00" in self.sample_id:
            raise ValueError("sample_id must be a non-empty bounded string")
        if len(self.prompt) > _MAX_GENERATED_TEXT or "\x00" in self.prompt:
            raise ValueError("prompt exceeds verification limits")
        if self.generated_text is not None and len(self.generated_text) > _MAX_GENERATED_TEXT:
            raise ValueError("generated text exceeds verification limits")
        if len(self.generated_token_ids) > 1_000_000 or any(
            isinstance(token, bool) or token < 0 for token in self.generated_token_ids
        ):
            raise ValueError("generated token IDs exceed verification limits")
        if len(self.values) > 10_000 or any(not isinstance(key, str) for key in self.values):
            raise ValueError("record values exceed verification limits")

    @property
    def prompt_hash(self) -> str:
        return sha256_bytes(self.prompt.encode("utf-8"))

    @property
    def output_hash(self) -> str | None:
        if self.generated_text is None:
            return None
        return sha256_bytes(self.generated_text.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class PromptMetric:
    sample_id: str
    prompt_hash: str
    output_hash: str | None
    outcome: VerificationOutcome
    metric: str
    value: MetricValue = None
    margin: float | None = None
    message: str | None = None
    diagnostics: Mapping[str, MetricValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "sample_id": self.sample_id,
            "prompt_hash": self.prompt_hash,
            "outcome": self.outcome.value,
            "metric": self.metric,
            "value": self.value,
            "diagnostics": dict(sorted(self.diagnostics.items())),
        }
        if self.output_hash is not None:
            result["output_hash"] = self.output_hash
        if self.margin is not None:
            result["margin"] = self.margin
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass(frozen=True, slots=True)
class AssertionEvaluation:
    assertion_id: str
    assertion_type: AssertionType
    outcome: VerificationOutcome
    metric: str
    value: float | None
    margin: float | None
    prompt_metrics: tuple[PromptMetric, ...]
    confidence_interval: ConfidenceInterval | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is VerificationOutcome.PASS and self.margin is not None and self.margin < 0:
            raise ValueError("passing assertion cannot have a negative margin")
        if (
            self.outcome is VerificationOutcome.FAIL
            and self.margin is not None
            and self.margin >= 0
        ):
            raise ValueError("failing assertion must have a negative margin")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "assertion_id": self.assertion_id,
            "assertion_type": self.assertion_type.value,
            "outcome": self.outcome.value,
            "metric": self.metric,
            "value": self.value,
            "margin": self.margin,
            "prompt_metrics": [item.to_dict() for item in self.prompt_metrics],
        }
        if self.confidence_interval is not None:
            result["confidence_interval"] = self.confidence_interval.to_dict()
        if self.message is not None:
            result["message"] = self.message
        return result


class _UnsupportedMetric(RuntimeError):
    pass


class _InvalidMetric(RuntimeError):
    pass


def _lookup(record: EvaluationRecord, assertion: VerificationAssertion, name: str) -> object:
    if name in record.values:
        return record.values[name]
    return assertion.option(name)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _UnsupportedMetric(f"{name} is not available as a numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise _InvalidMetric(f"{name} is non-finite")
    return result


def _option_float(
    assertion: VerificationAssertion, name: str, default: float | None = None
) -> float | None:
    value = assertion.option(name, default)
    if value is None:
        return None
    return _finite_float(value, name)


def _option_bool(assertion: VerificationAssertion, name: str, default: bool) -> bool:
    value = assertion.option(name, default)
    if not isinstance(value, bool):
        raise _InvalidMetric(f"{name} must be boolean")
    return value


def _text(record: EvaluationRecord) -> str:
    if record.generated_text is None:
        raise _UnsupportedMetric("generated_text is required")
    return record.generated_text


def _prompt_metric(
    record: EvaluationRecord,
    *,
    outcome: VerificationOutcome,
    metric: str,
    value: MetricValue = None,
    margin: float | None = None,
    message: str | None = None,
    diagnostics: Mapping[str, MetricValue] | None = None,
) -> PromptMetric:
    return PromptMetric(
        sample_id=record.sample_id,
        prompt_hash=record.prompt_hash,
        output_hash=record.output_hash,
        outcome=outcome,
        metric=metric,
        value=value,
        margin=margin,
        message=message,
        diagnostics=diagnostics or {},
    )


def _unsupported(
    assertion: VerificationAssertion,
    records: Sequence[EvaluationRecord],
    reason: str,
) -> AssertionEvaluation:
    return AssertionEvaluation(
        assertion_id=assertion.id,
        assertion_type=assertion.type,
        outcome=VerificationOutcome.UNSUPPORTED,
        metric=assertion.type.value,
        value=None,
        margin=None,
        prompt_metrics=tuple(
            _prompt_metric(
                record,
                outcome=VerificationOutcome.UNSUPPORTED,
                metric=assertion.type.value,
                message=reason,
            )
            for record in records
        ),
        message=reason,
    )


def _inconclusive(
    assertion: VerificationAssertion,
    records: Sequence[EvaluationRecord],
    reason: str,
) -> AssertionEvaluation:
    return AssertionEvaluation(
        assertion_id=assertion.id,
        assertion_type=assertion.type,
        outcome=VerificationOutcome.INCONCLUSIVE,
        metric=assertion.type.value,
        value=None,
        margin=None,
        prompt_metrics=tuple(
            _prompt_metric(
                record,
                outcome=VerificationOutcome.INCONCLUSIVE,
                metric=assertion.type.value,
                message=reason,
            )
            for record in records
        ),
        message=reason,
    )


def _confidence(
    values: Sequence[float], statistics: StatisticsPolicy | None
) -> ConfidenceInterval | None:
    if statistics is None:
        return None
    return bootstrap_mean_interval(
        values,
        confidence_level=statistics.confidence_level,
        samples=statistics.bootstrap_samples,
        seed=statistics.bootstrap_seed,
    )


def _binary_evaluation(
    assertion: VerificationAssertion,
    records: Sequence[EvaluationRecord],
    results: Sequence[tuple[bool, MetricValue, float | None, Mapping[str, MetricValue]]],
    *,
    metric: str,
    statistics: StatisticsPolicy | None,
) -> AssertionEvaluation:
    if not records:
        return _inconclusive(assertion, records, "assertion source produced no records")
    if len(records) != len(results):
        raise ValueError("record and result counts differ")
    threshold = _option_float(assertion, "minimum_pass_rate", 1.0)
    assert threshold is not None
    pass_values = [1.0 if passed else 0.0 for passed, _, _, _ in results]
    pass_rate = sum(pass_values) / len(pass_values)
    margin = pass_rate - threshold
    outcome = VerificationOutcome.PASS if margin >= 0.0 else VerificationOutcome.FAIL
    prompt_metrics = tuple(
        _prompt_metric(
            record,
            outcome=VerificationOutcome.PASS if result[0] else VerificationOutcome.FAIL,
            metric=metric,
            value=result[1],
            margin=result[2] if result[2] is not None else (1.0 if result[0] else -1.0),
            diagnostics=result[3],
        )
        for record, result in zip(records, results, strict=True)
    )
    return AssertionEvaluation(
        assertion_id=assertion.id,
        assertion_type=assertion.type,
        outcome=outcome,
        metric=metric,
        value=pass_rate,
        margin=margin,
        prompt_metrics=prompt_metrics,
        confidence_interval=_confidence(pass_values, statistics),
    )


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise _InvalidMetric("cannot compute a quantile of no values")
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def _continuous_evaluation(
    assertion: VerificationAssertion,
    records: Sequence[EvaluationRecord],
    values: Sequence[float],
    *,
    metric: str,
    statistics: StatisticsPolicy | None,
) -> AssertionEvaluation:
    if not values:
        return _inconclusive(assertion, records, "assertion source produced no records")
    conditions: list[tuple[str, float, float]] = []
    minimum = _option_float(assertion, "minimum")
    maximum = _option_float(assertion, "maximum")
    maximum_mean = _option_float(assertion, "maximum_mean")
    maximum_item = _option_float(assertion, "maximum_item")
    if minimum is not None:
        conditions.append(("minimum", min(values), min(values) - minimum))
    if maximum is not None:
        conditions.append(("maximum", max(values), maximum - max(values)))
    if maximum_mean is not None:
        observed_mean = sum(values) / len(values)
        conditions.append(("maximum_mean", observed_mean, maximum_mean - observed_mean))
    if maximum_item is not None:
        conditions.append(("maximum_item", max(values), maximum_item - max(values)))
    quantile_spec = assertion.option("maximum_quantile")
    if isinstance(quantile_spec, Mapping):
        q = _finite_float(quantile_spec.get("q"), "maximum_quantile.q")
        limit = _finite_float(quantile_spec.get("value"), "maximum_quantile.value")
        observed = _quantile(values, q)
        conditions.append((f"maximum_quantile_{q:g}", observed, limit - observed))
    if not conditions:
        raise _UnsupportedMetric("assertion has no supported numeric acceptance threshold")
    margin = min(item[2] for item in conditions)
    outcome = VerificationOutcome.PASS if margin >= 0.0 else VerificationOutcome.FAIL
    per_item_limit = maximum_item if maximum_item is not None else maximum
    item_minimum = minimum
    prompt_metrics: list[PromptMetric] = []
    for record, value in zip(records, values, strict=True):
        margins = []
        if per_item_limit is not None:
            margins.append(per_item_limit - value)
        if item_minimum is not None:
            margins.append(value - item_minimum)
        item_margin = min(margins) if margins else None
        item_outcome = (
            VerificationOutcome.FAIL
            if item_margin is not None and item_margin < 0.0
            else VerificationOutcome.PASS
        )
        prompt_metrics.append(
            _prompt_metric(
                record,
                outcome=item_outcome,
                metric=metric,
                value=value,
                margin=item_margin,
            )
        )
    return AssertionEvaluation(
        assertion_id=assertion.id,
        assertion_type=assertion.type,
        outcome=outcome,
        metric=metric,
        value=sum(values) / len(values),
        margin=margin,
        prompt_metrics=tuple(prompt_metrics),
        confidence_interval=_confidence(values, statistics),
        message="; ".join(f"{name}={observed:g}" for name, observed, _ in conditions),
    )


def _normalized(
    text: str,
    *,
    case_sensitive: bool,
    form: NormalizationForm = "NFKC",
) -> str:
    normalized = " ".join(unicodedata.normalize(form, text).split())
    return normalized if case_sensitive else normalized.casefold()


def _exact_results(
    assertion: VerificationAssertion,
    records: Sequence[EvaluationRecord],
    *,
    normalized: bool,
) -> list[BinaryResult]:
    results: list[BinaryResult] = []
    for record in records:
        output = _text(record)
        expected = _lookup(record, assertion, "expected")
        if not isinstance(expected, str):
            raise _UnsupportedMetric("expected output is missing")
        case_sensitive = _option_bool(assertion, "case_sensitive", True)
        if normalized:
            form = assertion.option("unicode_normalization", "NFKC")
            if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
                raise _InvalidMetric("unsupported Unicode normalization form")
            normalization_form = cast(NormalizationForm, form)
            actual_value = _normalized(
                output, case_sensitive=case_sensitive, form=normalization_form
            )
            expected_value = _normalized(
                expected, case_sensitive=case_sensitive, form=normalization_form
            )
        elif case_sensitive:
            actual_value, expected_value = output, expected
        else:
            actual_value, expected_value = output.casefold(), expected.casefold()
        passed = actual_value == expected_value
        results.append((passed, passed, 1.0 if passed else -1.0, {}))
    return results


def _safe_pattern(pattern: object) -> re.Pattern[str]:
    if not isinstance(pattern, str) or not pattern or len(pattern) > 512:
        raise _UnsupportedMetric("a bounded regular-expression pattern is required")
    if re.search(r"\\[1-9]|\(\?[=!<]|\(\?P|\(\?#|\(\?>", pattern):
        raise _UnsupportedMetric("pattern uses unsupported advanced regular-expression features")
    if re.search(r"(?:\*|\+|\?|\{[^}]*\})\s*(?:\*|\+|\?|\{)", pattern):
        raise _UnsupportedMetric("pattern uses repeated quantifiers")
    if re.search(r"(?:\*|\+|\?|\{[^}]*\})\)(?:\*|\+|\?|\{)", pattern):
        raise _UnsupportedMetric("pattern uses a quantified group containing a quantifier")
    try:
        return re.compile(pattern)
    except re.error as error:
        raise _UnsupportedMetric(f"invalid regular expression: {error}") from error


def _regex_results(
    assertion: VerificationAssertion, records: Sequence[EvaluationRecord]
) -> list[BinaryResult]:
    pattern_value = assertion.option("pattern")
    if not _option_bool(assertion, "case_sensitive", True) and isinstance(pattern_value, str):
        pattern = _safe_pattern(f"(?i:{pattern_value})")
    else:
        pattern = _safe_pattern(pattern_value)
    full = _option_bool(assertion, "full_match", False)
    results: list[BinaryResult] = []
    for record in records:
        output = _text(record)
        match = pattern.fullmatch(output) if full else pattern.search(output)
        passed = match is not None
        results.append((passed, passed, 1.0 if passed else -1.0, {}))
    return results


class _SchemaUnsupported(RuntimeError):
    pass


def _json_type_matches(instance: object, expected: str) -> bool:
    return {
        "null": instance is None,
        "boolean": isinstance(instance, bool),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "number": isinstance(instance, int | float)
        and not isinstance(instance, bool)
        and math.isfinite(float(instance)),
        "string": isinstance(instance, str),
        "array": isinstance(instance, list),
        "object": isinstance(instance, Mapping),
    }.get(expected, False)


def _schema_errors(
    instance: object,
    schema: Mapping[str, object],
    path: str,
    depth: int,
) -> list[str]:
    if depth > 64:
        raise _SchemaUnsupported("JSON schema exceeds supported nesting depth")
    allowed = {
        "$schema",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
    unknown = set(schema) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise _SchemaUnsupported(f"unsupported JSON Schema keyword(s): {names}")
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        types = [expected_type] if isinstance(expected_type, str) else expected_type
        if (
            not isinstance(types, list | tuple)
            or not types
            or not all(isinstance(item, str) for item in types)
        ):
            raise _SchemaUnsupported("schema type must be a string or array of strings")
        if not any(_json_type_matches(instance, cast(str, item)) for item in types):
            errors.append(f"{path}: expected type {types!r}")
            return errors
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: does not equal const")
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list | tuple):
            raise _SchemaUnsupported("schema enum must be an array")
        if not any(instance == candidate for candidate in enum):
            errors.append(f"{path}: value is not in enum")
    if isinstance(instance, Mapping):
        required = schema.get("required", ())
        valid_required = isinstance(required, list | tuple) and all(
            isinstance(item, str) for item in required
        )
        if not valid_required:
            raise _SchemaUnsupported("schema required must be an array of strings")
        required_keys = cast(Sequence[str], required)
        for key in required_keys:
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise _SchemaUnsupported("schema properties must be an object")
        for key, child_schema in properties.items():
            if not isinstance(key, str) or not isinstance(child_schema, Mapping):
                raise _SchemaUnsupported("property schemas must be objects")
            if key in instance:
                child_errors = _schema_errors(
                    instance[key], child_schema, f"{path}.{key}", depth + 1
                )
                errors.extend(child_errors)
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}: additional property {key!r} is forbidden")
        elif isinstance(additional, Mapping):
            for key in instance:
                if key not in properties:
                    errors.extend(
                        _schema_errors(instance[key], additional, f"{path}.{key}", depth + 1)
                    )
        elif additional is not True:
            raise _SchemaUnsupported("additionalProperties must be boolean or a schema")
    if isinstance(instance, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(instance) < minimum:
            errors.append(f"{path}: has fewer than minItems")
        if isinstance(maximum, int) and len(instance) > maximum:
            errors.append(f"{path}: has more than maxItems")
        if schema.get("uniqueItems") is True:
            encoded = [json.dumps(item, sort_keys=True, allow_nan=False) for item in instance]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items are not unique")
        items = schema.get("items")
        if items is not None:
            if not isinstance(items, Mapping):
                raise _SchemaUnsupported("schema items must be an object")
            for index, item in enumerate(instance):
                errors.extend(_schema_errors(item, items, f"{path}[{index}]", depth + 1))
    if isinstance(instance, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(instance) < minimum:
            errors.append(f"{path}: shorter than minLength")
        if isinstance(maximum, int) and len(instance) > maximum:
            errors.append(f"{path}: longer than maxLength")
        if "pattern" in schema and _safe_pattern(schema["pattern"]).search(instance) is None:
            errors.append(f"{path}: does not match pattern")
    if isinstance(instance, int | float) and not isinstance(instance, bool):
        numeric = float(instance)
        numeric_bounds = {
            name: _finite_float(schema[name], name)
            for name in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
            if name in schema
        }
        if "minimum" in numeric_bounds and numeric < numeric_bounds["minimum"]:
            errors.append(f"{path}: violates minimum")
        if "maximum" in numeric_bounds and numeric > numeric_bounds["maximum"]:
            errors.append(f"{path}: violates maximum")
        if "exclusiveMinimum" in numeric_bounds and numeric <= numeric_bounds["exclusiveMinimum"]:
            errors.append(f"{path}: violates exclusiveMinimum")
        if "exclusiveMaximum" in numeric_bounds and numeric >= numeric_bounds["exclusiveMaximum"]:
            errors.append(f"{path}: violates exclusiveMaximum")
    return errors


def _parse_json_output(text: str) -> object:
    try:
        return loads_data(text, format="json", limits=_JSON_OUTPUT_LIMITS)
    except (ContractSyntaxError, ValueError) as error:
        raise _InvalidMetric(str(error)) from error


def _json_results(
    assertion: VerificationAssertion,
    records: Sequence[EvaluationRecord],
    *,
    schemas: Mapping[str, Mapping[str, object]],
) -> list[BinaryResult]:
    schema: Mapping[str, object] | None = None
    if assertion.type is AssertionType.JSON_SCHEMA:
        raw_schema = assertion.option("schema")
        schema_file = assertion.option("schema_file")
        if isinstance(raw_schema, Mapping):
            schema = cast(Mapping[str, object], raw_schema)
        elif isinstance(schema_file, str):
            schema = schemas.get(schema_file)
            if schema is None:
                raise _UnsupportedMetric(
                    "schema_file must be resolved by the trusted verification loader"
                )
        else:
            raise _UnsupportedMetric("JSON schema is unavailable")
    results: list[BinaryResult] = []
    for record in records:
        try:
            instance = _parse_json_output(_text(record))
            errors = _schema_errors(instance, schema, "$", 0) if schema is not None else []
            passed = not errors
            results.append(
                (
                    passed,
                    passed,
                    1.0 if passed else -1.0,
                    {"parser": "strict_json", "error": errors[0] if errors else None},
                )
            )
        except _InvalidMetric as error:
            results.append((False, False, -1.0, {"parser": "strict_json", "error": str(error)}))
    return results


def _logits_2d(tensor: torch.Tensor | None, name: str) -> torch.Tensor:
    if tensor is None:
        raise _UnsupportedMetric(f"{name} is required")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 2:
        raise _InvalidMetric(f"{name} must have shape [time, vocabulary]")
    if not bool(torch.isfinite(tensor).all().item()):
        raise _InvalidMetric(f"{name} contains non-finite values")
    return tensor


def _ids_1d(tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        raise _UnsupportedMetric("input_ids are required")
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1 or tensor.shape[0] < 2:
        raise _InvalidMetric("input_ids must contain at least two tokens")
    return tensor.to(dtype=torch.long)


def _sequence_log_probability(record: EvaluationRecord, *, normalize: bool) -> float:
    supplied = record.values.get("sequence_log_probability")
    if supplied is not None:
        return _finite_float(supplied, "sequence_log_probability")
    logits = _logits_2d(record.logits, "logits")
    ids = _ids_1d(record.input_ids).to(device=logits.device)
    if logits.shape[0] < ids.shape[0] - 1:
        raise _InvalidMetric("logits do not cover every next-token target")
    next_logits = logits[: ids.shape[0] - 1]
    targets = ids[1:]
    if int(targets.max().item()) >= next_logits.shape[-1] or int(targets.min().item()) < 0:
        raise _InvalidMetric("input token ID is outside the logit vocabulary")
    scores = functional.log_softmax(next_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    value = scores.mean() if normalize else scores.sum()
    return float(value.detach().cpu())


def _token_log_probability(record: EvaluationRecord, assertion: VerificationAssertion) -> float:
    supplied = record.values.get("token_log_probability")
    if supplied is not None:
        return _finite_float(supplied, "token_log_probability")
    logits = _logits_2d(record.logits, "logits")
    position_raw = _lookup(record, assertion, "position")
    position = -1 if position_raw is None else int(_finite_float(position_raw, "position"))
    token_raw = _lookup(record, assertion, "token_id")
    if token_raw is None:
        ids = _ids_1d(record.input_ids).to(device=logits.device)
        target_index = position if position >= 0 else ids.shape[0] + position
        if not 1 <= target_index < ids.shape[0]:
            raise _InvalidMetric("target token position is outside the causal sequence")
        logit_index = target_index - 1
        token_id = int(ids[target_index].item())
    else:
        token_id = int(_finite_float(token_raw, "token_id"))
        logit_index = position if position >= 0 else logits.shape[0] + position
    if not 0 <= logit_index < logits.shape[0] or not 0 <= token_id < logits.shape[-1]:
        raise _InvalidMetric("token or logit position is out of range")
    return float(functional.log_softmax(logits[logit_index], dim=-1)[token_id].detach().cpu())


def _score_mapping(record: EvaluationRecord, name: str) -> Mapping[str, object]:
    value = record.values.get(name)
    if not isinstance(value, Mapping):
        raise _UnsupportedMetric(f"{name} mapping is required")
    return cast(Mapping[str, object], value)


def _margin_results(
    assertion: VerificationAssertion, records: Sequence[EvaluationRecord]
) -> list[BinaryResult]:
    threshold = _option_float(assertion, "minimum_margin", 0.0)
    assert threshold is not None
    results: list[BinaryResult] = []
    for record in records:
        scores = _score_mapping(record, "sequence_log_probabilities")
        preferred = _lookup(record, assertion, "preferred")
        dispreferred = _lookup(record, assertion, "dispreferred")
        if not isinstance(preferred, str) or not isinstance(dispreferred, str):
            raise _UnsupportedMetric("preferred and dispreferred sequences are required")
        if preferred not in scores or dispreferred not in scores:
            raise _UnsupportedMetric("sequence scores do not include declared alternatives")
        margin = _finite_float(scores[preferred], preferred) - _finite_float(
            scores[dispreferred], dispreferred
        )
        results.append((margin >= threshold, margin, margin - threshold, {}))
    return results


def _choice_results(
    assertion: VerificationAssertion, records: Sequence[EvaluationRecord]
) -> list[BinaryResult]:
    threshold = _option_float(assertion, "minimum_margin", 0.0)
    assert threshold is not None
    results: list[BinaryResult] = []
    for record in records:
        scores = _score_mapping(record, "choice_log_probabilities")
        correct = _lookup(record, assertion, "correct_choice")
        choices_raw = _lookup(record, assertion, "choices")
        choices = tuple(scores)
        if choices_raw is not None:
            choices = tuple(cast(Sequence[str], choices_raw))
        if not isinstance(correct, str) or correct not in scores:
            raise _UnsupportedMetric("correct choice score is unavailable")
        alternatives = [
            _finite_float(scores[choice], choice)
            for choice in choices
            if isinstance(choice, str) and choice != correct and choice in scores
        ]
        if not alternatives:
            raise _UnsupportedMetric("at least one alternative choice score is required")
        margin = _finite_float(scores[correct], correct) - max(alternatives)
        results.append((margin >= threshold, margin, margin - threshold, {}))
    return results


def _kl_values(
    records: Sequence[EvaluationRecord], *, reference_name: str, temperature: float
) -> list[float]:
    values = []
    for record in records:
        supplied = record.values.get(f"{reference_name}_kl")
        if supplied is not None:
            values.append(_finite_float(supplied, f"{reference_name}_kl"))
            continue
        student = _logits_2d(record.logits, "logits")
        if reference_name == "reference":
            reference_tensor = record.reference_logits
        else:
            reference_tensor = record.base_logits
        reference = _logits_2d(reference_tensor, f"{reference_name}_logits").to(
            device=student.device, dtype=student.dtype
        )
        if student.shape != reference.shape:
            raise _InvalidMetric("student and reference logits have different shapes")
        student_log = functional.log_softmax(student / temperature, dim=-1)
        reference_log = functional.log_softmax(reference / temperature, dim=-1)
        divergence = (reference_log.exp() * (reference_log - student_log)).sum(dim=-1).mean() * (
            temperature**2
        )
        values.append(float(divergence.detach().cpu()))
    return values


def _perplexity_values(records: Sequence[EvaluationRecord]) -> list[float]:
    values = []
    for record in records:
        supplied = record.values.get("perplexity")
        if supplied is not None:
            values.append(_finite_float(supplied, "perplexity"))
            continue
        negative_log_likelihood = -_sequence_log_probability(record, normalize=True)
        if negative_log_likelihood > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            raise _InvalidMetric("perplexity overflows finite floating point")
        values.append(math.exp(negative_log_likelihood))
    return values


def _free_generation_results(
    assertion: VerificationAssertion, records: Sequence[EvaluationRecord]
) -> list[BinaryResult]:
    match_type = assertion.option("match_type", "exact")
    if match_type == "regex":
        return _regex_results(assertion, records)
    results: list[BinaryResult] = []
    for record in records:
        output = _text(record)
        expected = _lookup(record, assertion, "expected")
        if not isinstance(expected, str):
            raise _UnsupportedMetric("expected free-generation output is missing")
        case_sensitive = _option_bool(assertion, "case_sensitive", True)
        actual_cmp = output if case_sensitive else output.casefold()
        expected_cmp = expected if case_sensitive else expected.casefold()
        if match_type == "exact":
            passed = actual_cmp == expected_cmp
        elif match_type == "normalized":
            passed = _normalized(output, case_sensitive=case_sensitive) == _normalized(
                expected, case_sensitive=case_sensitive
            )
        elif match_type == "contains":
            passed = expected_cmp in actual_cmp
        else:
            raise _UnsupportedMetric(f"unsupported free-generation match_type {match_type!r}")
        results.append((passed, passed, 1.0 if passed else -1.0, {}))
    return results


def evaluate_assertion(
    assertion: VerificationAssertion,
    records: Sequence[EvaluationRecord],
    *,
    statistics: StatisticsPolicy | None = None,
    schemas: Mapping[str, Mapping[str, object]] | None = None,
) -> AssertionEvaluation:
    """Execute one assertion and preserve every prompt-level outcome."""

    record_tuple = tuple(records)
    if not record_tuple:
        return _inconclusive(assertion, record_tuple, "assertion source produced no records")
    try:
        if assertion.type is AssertionType.EXACT_MATCH:
            results = _exact_results(assertion, record_tuple, normalized=False)
            return _binary_evaluation(
                assertion, record_tuple, results, metric="exact_match", statistics=statistics
            )
        if assertion.type is AssertionType.NORMALIZED_EXACT_MATCH:
            results = _exact_results(assertion, record_tuple, normalized=True)
            return _binary_evaluation(
                assertion,
                record_tuple,
                results,
                metric="normalized_exact_match",
                statistics=statistics,
            )
        if assertion.type is AssertionType.REGULAR_EXPRESSION:
            return _binary_evaluation(
                assertion,
                record_tuple,
                _regex_results(assertion, record_tuple),
                metric="regular_expression",
                statistics=statistics,
            )
        if assertion.type in {AssertionType.JSON_PARSE, AssertionType.JSON_SCHEMA}:
            return _binary_evaluation(
                assertion,
                record_tuple,
                _json_results(assertion, record_tuple, schemas=schemas or {}),
                metric=assertion.type.value,
                statistics=statistics,
            )
        if assertion.type is AssertionType.FREE_GENERATION_MATCH:
            return _binary_evaluation(
                assertion,
                record_tuple,
                _free_generation_results(assertion, record_tuple),
                metric="free_generation_match",
                statistics=statistics,
            )
        if assertion.type is AssertionType.SEQUENCE_MARGIN:
            return _binary_evaluation(
                assertion,
                record_tuple,
                _margin_results(assertion, record_tuple),
                metric="sequence_margin",
                statistics=statistics,
            )
        if assertion.type is AssertionType.MULTIPLE_CHOICE_MARGIN:
            return _binary_evaluation(
                assertion,
                record_tuple,
                _choice_results(assertion, record_tuple),
                metric="multiple_choice_margin",
                statistics=statistics,
            )
        if assertion.type is AssertionType.TOKEN_LOG_PROBABILITY:
            values = [_token_log_probability(record, assertion) for record in record_tuple]
            return _continuous_evaluation(
                assertion,
                record_tuple,
                values,
                metric="token_log_probability",
                statistics=statistics,
            )
        if assertion.type is AssertionType.SEQUENCE_LOG_PROBABILITY:
            normalize = _option_bool(assertion, "normalize", False)
            values = [
                _sequence_log_probability(record, normalize=normalize) for record in record_tuple
            ]
            return _continuous_evaluation(
                assertion,
                record_tuple,
                values,
                metric="sequence_log_probability",
                statistics=statistics,
            )
        if assertion.type in {AssertionType.REFERENCE_KL, AssertionType.BASE_KL}:
            temperature = _option_float(assertion, "temperature", 1.0)
            assert temperature is not None
            reference_name = "reference" if assertion.type is AssertionType.REFERENCE_KL else "base"
            values = _kl_values(
                record_tuple, reference_name=reference_name, temperature=temperature
            )
            return _continuous_evaluation(
                assertion,
                record_tuple,
                values,
                metric=f"{reference_name}_kl",
                statistics=statistics,
            )
        if assertion.type is AssertionType.GENERATION_LENGTH:
            unit = assertion.option("unit", "tokens")
            values = []
            for record in record_tuple:
                if unit == "tokens":
                    if not record.generated_token_ids:
                        message = "generated token IDs are required for token length"
                        raise _UnsupportedMetric(message)
                    values.append(float(len(record.generated_token_ids)))
                elif unit == "characters":
                    values.append(float(len(_text(record))))
                elif unit == "words":
                    values.append(float(len(_text(record).split())))
                else:
                    raise _UnsupportedMetric(f"unsupported generation length unit {unit!r}")
            return _continuous_evaluation(
                assertion,
                record_tuple,
                values,
                metric=f"generation_length_{unit}",
                statistics=statistics,
            )
        if assertion.type is AssertionType.PERPLEXITY:
            return _continuous_evaluation(
                assertion,
                record_tuple,
                _perplexity_values(record_tuple),
                metric="perplexity",
                statistics=statistics,
            )
        return _unsupported(assertion, record_tuple, "assertion type is not implemented")
    except (_SchemaUnsupported, _UnsupportedMetric) as error:
        return _unsupported(assertion, record_tuple, str(error))
    except (_InvalidMetric, RuntimeError, ValueError) as error:
        return _inconclusive(assertion, record_tuple, str(error))


__all__ = [
    "AssertionEvaluation",
    "EvaluationRecord",
    "MetricValue",
    "PromptMetric",
    "evaluate_assertion",
]
