"""Strict, resource-bounded YAML/JSON parser for Behavior Contract v1."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.tokens import AliasToken, AnchorToken, TagToken

from modelpact.contracts.ast import (
    AssertionType,
    BehaviorContract,
    CompileObjective,
    GenerationMode,
    GenerationPolicy,
    HoldoutPolicy,
    JsonValue,
    ModelRequirements,
    ObjectiveType,
    StatisticsPolicy,
    UnsealPolicy,
    VerificationAssertion,
)
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.paths import resolve_inside


@dataclass(frozen=True, slots=True)
class ContractLimits:
    max_bytes: int = 2 * 1024 * 1024
    max_depth: int = 32
    max_nodes: int = 100_000
    max_string_length: int = 1_000_000
    max_object_keys: int = 10_000
    max_objectives: int = 10_000
    max_assertions: int = 50_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_LIMITS = ContractLimits()


class ContractError(ValueError):
    """Base class for untrusted contract-data failures."""


class ContractSyntaxError(ContractError):
    pass


class ContractValidationError(ContractError):
    pass


class ContractResourceLimitError(ContractError):
    pass


class _StrictLoader(yaml.SafeLoader):
    pass


# Copy resolver tables before modifying them: mutating the inherited mappings
# would alter global SafeLoader behavior in the hosting process.
_StrictLoader.yaml_implicit_resolvers = {
    key: list(value) for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _key, _resolvers in tuple(_StrictLoader.yaml_implicit_resolvers.items()):
    _StrictLoader.yaml_implicit_resolvers[_key] = [
        item
        for item in _resolvers
        if item[0] not in {"tag:yaml.org,2002:timestamp", "tag:yaml.org,2002:bool"}
    ]
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _construct_mapping_no_duplicates(
    loader: _StrictLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping key is not scalar",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_duplicates,
)


def _fail(path: str, message: str) -> NoReturn:
    raise ContractValidationError(f"{path}: {message}")


def _reject_constant(value: str) -> NoReturn:
    raise ContractSyntaxError(f"non-finite JSON number {value!r} is not permitted")


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractSyntaxError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_document_shape(value: object, limits: ContractLimits) -> None:
    nodes = 0
    stack: list[tuple[object, int, str]] = [(value, 0, "$")]
    active: set[int] = set()
    # Parsed YAML aliases are rejected before construction, but the cycle check
    # is retained for callers of the generic JSON-document validator.
    while stack:
        item, depth, path = stack.pop()
        nodes += 1
        if nodes > limits.max_nodes:
            raise ContractResourceLimitError(f"document exceeds {limits.max_nodes} nodes")
        if depth > limits.max_depth:
            raise ContractResourceLimitError(
                f"{path}: document exceeds nesting depth {limits.max_depth}"
            )
        if item is None or isinstance(item, bool | int):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                _fail(path, "non-finite numbers are not permitted")
            continue
        if isinstance(item, str):
            if len(item) > limits.max_string_length:
                raise ContractResourceLimitError(
                    f"{path}: string exceeds {limits.max_string_length} characters"
                )
            if "\x00" in item:
                _fail(path, "NUL characters are not permitted")
            continue
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                _fail(path, "cyclic data is not permitted")
            active.add(identity)
            if len(item) > limits.max_object_keys:
                raise ContractResourceLimitError(
                    f"{path}: object exceeds {limits.max_object_keys} keys"
                )
            for key, child in item.items():
                if not isinstance(key, str):
                    _fail(path, "object keys must be strings")
                if len(key) > limits.max_string_length:
                    raise ContractResourceLimitError(
                        f"{path}: object key exceeds {limits.max_string_length} characters"
                    )
                if "\x00" in key:
                    _fail(path, "NUL characters are not permitted in object keys")
                stack.append((child, depth + 1, f"{path}.{key}"))
            # Since YAML aliases are disallowed, seeing an object again is not
            # useful; remove it after its children have been scheduled.
            active.remove(identity)
            continue
        if isinstance(item, Sequence) and not isinstance(item, bytes | bytearray):
            identity = id(item)
            if identity in active:
                _fail(path, "cyclic data is not permitted")
            active.add(identity)
            for index, child in enumerate(item):
                stack.append((child, depth + 1, f"{path}[{index}]"))
            active.remove(identity)
            continue
        _fail(path, f"unsupported data type {type(item).__name__}")


def validate_data_shape(value: object, *, limits: ContractLimits = DEFAULT_LIMITS) -> None:
    """Apply the shared hostile-data resource policy to an in-memory value."""

    _validate_document_shape(value, limits)


def loads_data(
    text: str | bytes,
    *,
    format: str,
    limits: ContractLimits = DEFAULT_LIMITS,
    require_canonical: bool = False,
) -> JsonValue:
    """Parse a bounded JSON or safe-YAML document without interpreting a schema.

    When ``require_canonical`` is true, JSON input must be exactly the ModelPact
    canonical encoding, optionally followed by one LF written by artifact
    writers.  This mode deliberately rejects BOMs, CRLF, insignificant
    whitespace, alternate escapes, and non-canonical number spellings instead
    of silently normalizing an untrusted content-addressed record.
    """

    raw = text.encode("utf-8") if isinstance(text, str) else text
    if len(raw) > limits.max_bytes:
        raise ContractResourceLimitError(f"document exceeds {limits.max_bytes} bytes")
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ContractSyntaxError("document must be UTF-8") from error
    normalized_format = format.lower().lstrip(".")
    try:
        if normalized_format == "json":
            value = json.loads(
                decoded,
                object_pairs_hook=_json_object,
                parse_constant=_reject_constant,
            )
        elif normalized_format in {"yaml", "yml"}:
            try:
                tokens = yaml.scan(decoded, Loader=_StrictLoader)
                for token in tokens:
                    if isinstance(token, AnchorToken | AliasToken | TagToken):
                        raise ContractSyntaxError(
                            "YAML anchors, aliases, and explicit tags are not permitted"
                        )
                # _StrictLoader subclasses SafeLoader and rejects tags/aliases above.
                value = yaml.load(decoded, Loader=_StrictLoader)  # noqa: S506
            except ContractSyntaxError:
                raise
            except yaml.YAMLError as error:
                raise ContractSyntaxError(f"malformed YAML: {error}") from error
        else:
            raise ContractSyntaxError("format must be json, yaml, or yml")
    except ContractError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ContractSyntaxError(f"malformed JSON: {error}") from error
    _validate_document_shape(value, limits)
    if require_canonical:
        if normalized_format != "json":
            raise ContractSyntaxError("canonical representation is defined only for JSON")
        canonical = canonical_dumps(value, max_depth=limits.max_depth).encode("utf-8")
        if raw not in {canonical, canonical + b"\n"}:
            raise ContractSyntaxError("JSON document is not in canonical ModelPact encoding")
    return cast(JsonValue, value)


def load_data_file(
    path: str | Path,
    *,
    limits: ContractLimits = DEFAULT_LIMITS,
    require_canonical: bool = False,
) -> JsonValue:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise ContractSyntaxError(f"cannot read {source}: {error}") from error
    if size > limits.max_bytes:
        raise ContractResourceLimitError(f"document exceeds {limits.max_bytes} bytes")
    suffix = source.suffix.lower().lstrip(".")
    if suffix not in {"json", "yaml", "yml"}:
        raise ContractSyntaxError("contract filename must end in .json, .yaml, or .yml")
    try:
        data = source.read_bytes()
    except OSError as error:
        raise ContractSyntaxError(f"cannot read {source}: {error}") from error
    return loads_data(
        data,
        format=suffix,
        limits=limits,
        require_canonical=require_canonical,
    )


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object, path: str, *, maximum: int) -> list[object]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    if len(value) > maximum:
        raise ContractResourceLimitError(f"{path}: array exceeds {maximum} entries")
    return value


def _string(value: object, path: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _required_string(value: object, path: str) -> str:
    result = _string(value, path)
    assert result is not None
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be finite")
    return result


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _known_fields(value: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail(path, "unknown field(s): " + ", ".join(unknown))


_OBJECTIVE_OPTIONS: dict[ObjectiveType, set[str]] = {
    ObjectiveType.TEACHER_CROSS_ENTROPY: {"temperature", "ignore_index", "causal_shift"},
    ObjectiveType.TEACHER_KL: {"temperature", "causal_shift"},
    ObjectiveType.PREFERRED_SEQUENCE_MARGIN: {"margin", "reduction"},
    ObjectiveType.BASE_KL: {"temperature", "causal_shift"},
    ObjectiveType.HIDDEN_STATE_MATCHING: {"activation_point", "metric"},
    ObjectiveType.ACTIVATION_DIRECTION: {
        "activation_point",
        "direction_key",
        "minimum_projection",
        "absolute",
    },
}

_COMMON_ASSERTION_OPTIONS = {
    "minimum_pass_rate",
    "input_hash",
    "prompt",
    "minimum",
    "maximum",
}
_ASSERTION_OPTIONS: dict[AssertionType, set[str]] = {
    AssertionType.TOKEN_LOG_PROBABILITY: {
        "token_id",
        "token",
        "position",
        "aggregation",
    },
    AssertionType.SEQUENCE_LOG_PROBABILITY: {"sequence", "normalize"},
    AssertionType.SEQUENCE_MARGIN: {"preferred", "dispreferred", "minimum_margin"},
    AssertionType.MULTIPLE_CHOICE_MARGIN: {
        "choices",
        "correct_choice",
        "minimum_margin",
    },
    AssertionType.EXACT_MATCH: {"expected", "case_sensitive"},
    AssertionType.NORMALIZED_EXACT_MATCH: {"expected", "unicode_normalization"},
    AssertionType.REGULAR_EXPRESSION: {"pattern", "full_match", "case_sensitive"},
    AssertionType.JSON_PARSE: set(),
    AssertionType.JSON_SCHEMA: {"schema", "schema_file"},
    AssertionType.FREE_GENERATION_MATCH: {
        "expected",
        "pattern",
        "match_type",
        "case_sensitive",
    },
    AssertionType.REFERENCE_KL: {
        "maximum_mean",
        "maximum_item",
        "maximum_quantile",
        "temperature",
    },
    AssertionType.BASE_KL: {
        "maximum_mean",
        "maximum_item",
        "maximum_quantile",
        "temperature",
    },
    AssertionType.GENERATION_LENGTH: {"unit"},
    AssertionType.PERPLEXITY: {"maximum_mean", "maximum_item"},
}


def _merge_options(
    value: Mapping[str, object],
    *,
    generic: set[str],
    allowed: set[str],
    path: str,
) -> dict[str, JsonValue]:
    top_options = set(value) - generic - {"parameters"}
    unknown = top_options - allowed
    if unknown:
        _fail(path, "unknown field(s): " + ", ".join(sorted(unknown)))
    result: dict[str, JsonValue] = {
        key: cast(JsonValue, item) for key, item in value.items() if key in allowed
    }
    if "parameters" in value:
        supplied = _mapping(value["parameters"], f"{path}.parameters")
        _known_fields(supplied, allowed, f"{path}.parameters")
        duplicates = set(result) & set(supplied)
        if duplicates:
            _fail(path, "option declared twice: " + ", ".join(sorted(duplicates)))
        result.update({key: cast(JsonValue, item) for key, item in supplied.items()})
    return result


def _validate_rate(options: Mapping[str, JsonValue], path: str) -> None:
    if "minimum_pass_rate" in options:
        rate = _number(options["minimum_pass_rate"], f"{path}.minimum_pass_rate")
        if not 0.0 <= rate <= 1.0:
            _fail(f"{path}.minimum_pass_rate", "must be in [0, 1]")


def _validate_bounds(options: Mapping[str, JsonValue], path: str) -> None:
    minimum = _number(options["minimum"], f"{path}.minimum") if "minimum" in options else None
    maximum = _number(options["maximum"], f"{path}.maximum") if "maximum" in options else None
    if minimum is not None and maximum is not None and minimum > maximum:
        _fail(path, "minimum cannot exceed maximum")


def _validate_regex(pattern: str, path: str) -> None:
    if len(pattern) > 512:
        _fail(path, "pattern exceeds 512 characters")
    # Python's backtracking engine has no portable timeout.  Contract v1 uses a
    # deliberately conservative subset for data supplied by untrusted bundles.
    if re.search(r"\\[1-9]|\(\?[=!<]|\(\?P|\(\?#|\(\?>", pattern):
        _fail(path, "backreferences, lookarounds, and special groups are unsupported")
    if re.search(r"(?:\*|\+|\?|\{[^}]*\})(?:\s*)(?:\*|\+|\?|\{)", pattern):
        _fail(path, "nested or repeated quantifiers are unsupported")
    if re.search(r"(?:\*|\+|\?|\{[^}]*\})\)(?:\*|\+|\?|\{)", pattern):
        _fail(path, "quantified groups containing quantifiers are unsupported")
    try:
        re.compile(pattern)
    except re.error as error:
        _fail(path, f"invalid regular expression: {error}")


def _parse_objective(value: object, path: str) -> CompileObjective:
    obj = _mapping(value, path)
    generic = {"id", "type", "source", "weight"}
    type_text = _required_string(obj.get("type"), f"{path}.type")
    try:
        objective_type = ObjectiveType(type_text)
    except ValueError:
        _fail(f"{path}.type", f"unknown objective type {type_text!r}")
    options = _merge_options(
        obj,
        generic=generic,
        allowed=_OBJECTIVE_OPTIONS[objective_type],
        path=path,
    )
    for key in ("temperature", "margin", "minimum_projection"):
        if key in options:
            number = _number(options[key], f"{path}.{key}")
            if key == "temperature" and number <= 0.0:
                _fail(f"{path}.{key}", "must be positive")
    if "ignore_index" in options:
        _integer(options["ignore_index"], f"{path}.ignore_index")
    if "causal_shift" in options:
        _boolean(options["causal_shift"], f"{path}.causal_shift")
    if "absolute" in options:
        _boolean(options["absolute"], f"{path}.absolute")
    if "metric" in options and options["metric"] not in {"mse", "cosine"}:
        _fail(f"{path}.metric", "must be mse or cosine")
    if "reduction" in options and options["reduction"] not in {"mean", "sum"}:
        _fail(f"{path}.reduction", "must be mean or sum")
    try:
        return CompileObjective(
            id=_required_string(obj.get("id"), f"{path}.id"),
            type=objective_type,
            source=_required_string(obj.get("source"), f"{path}.source"),
            weight=_number(obj.get("weight", 1.0), f"{path}.weight"),
            options=options,
        )
    except ValueError as error:
        _fail(path, str(error))


def _validate_assertion_options(
    assertion_type: AssertionType, options: dict[str, JsonValue], path: str
) -> None:
    _validate_rate(options, path)
    _validate_bounds(options, path)
    for key in ("minimum_margin", "temperature", "maximum_mean", "maximum_item"):
        if key in options:
            number = _number(options[key], f"{path}.{key}")
            if key.startswith("maximum_") and number < 0.0:
                _fail(f"{path}.{key}", "must be non-negative")
            if key == "temperature" and number <= 0.0:
                _fail(f"{path}.{key}", "must be positive")
    if "maximum_quantile" in options:
        quantile = _mapping(options["maximum_quantile"], f"{path}.maximum_quantile")
        _known_fields(quantile, {"q", "value"}, f"{path}.maximum_quantile")
        q = _number(quantile.get("q"), f"{path}.maximum_quantile.q")
        limit = _number(quantile.get("value"), f"{path}.maximum_quantile.value")
        if not 0.0 < q <= 1.0 or limit < 0.0:
            _fail(f"{path}.maximum_quantile", "q must be in (0, 1] and value non-negative")
    if assertion_type in {AssertionType.REFERENCE_KL, AssertionType.BASE_KL} and not any(
        key in options for key in ("maximum_mean", "maximum_item", "maximum_quantile")
    ):
        _fail(path, "KL assertions require at least one maximum threshold")
    if assertion_type is AssertionType.PERPLEXITY and not any(
        key in options for key in ("maximum_mean", "maximum_item", "maximum")
    ):
        _fail(path, "perplexity requires a maximum threshold")
    if assertion_type in {
        AssertionType.TOKEN_LOG_PROBABILITY,
        AssertionType.SEQUENCE_LOG_PROBABILITY,
    } and not any(key in options for key in ("minimum", "maximum")):
        _fail(path, "log-probability assertions require minimum or maximum")
    if assertion_type is AssertionType.GENERATION_LENGTH and not any(
        key in options for key in ("minimum", "maximum")
    ):
        _fail(path, "generation_length requires minimum or maximum")
    if "pattern" in options:
        pattern = _required_string(options["pattern"], f"{path}.pattern")
        _validate_regex(pattern, f"{path}.pattern")
    if assertion_type is AssertionType.REGULAR_EXPRESSION and "pattern" not in options:
        _fail(path, "regular_expression requires pattern")
    if assertion_type is AssertionType.JSON_SCHEMA:
        present = {key for key in ("schema", "schema_file") if key in options}
        if len(present) != 1:
            _fail(path, "json_schema requires exactly one of schema or schema_file")
        if "schema" in options:
            _mapping(options["schema"], f"{path}.schema")
        if "schema_file" in options:
            _required_string(options["schema_file"], f"{path}.schema_file")
    if "choices" in options:
        choices = _list(options["choices"], f"{path}.choices", maximum=10_000)
        if len(choices) < 2 or any(not isinstance(item, str) or not item for item in choices):
            _fail(f"{path}.choices", "must contain at least two non-empty strings")
    for key in ("case_sensitive", "full_match", "normalize"):
        if key in options:
            _boolean(options[key], f"{path}.{key}")
    if "unit" in options and options["unit"] not in {"tokens", "characters", "words"}:
        _fail(f"{path}.unit", "must be tokens, characters, or words")


def _parse_assertion(value: object, path: str) -> VerificationAssertion:
    obj = _mapping(value, path)
    generic = {"id", "type", "source"}
    type_text = _required_string(obj.get("type"), f"{path}.type")
    try:
        assertion_type = AssertionType(type_text)
    except ValueError:
        _fail(f"{path}.type", f"unknown assertion type {type_text!r}")
    allowed = _COMMON_ASSERTION_OPTIONS | _ASSERTION_OPTIONS[assertion_type]
    options = _merge_options(obj, generic=generic, allowed=allowed, path=path)
    _validate_assertion_options(assertion_type, options, path)
    try:
        return VerificationAssertion(
            id=_required_string(obj.get("id"), f"{path}.id"),
            type=assertion_type,
            source=_required_string(obj.get("source"), f"{path}.source"),
            options=options,
        )
    except ValueError as error:
        _fail(path, str(error))


def parse_contract(value: object, *, limits: ContractLimits = DEFAULT_LIMITS) -> BehaviorContract:
    _validate_document_shape(value, limits)
    root = _mapping(value, "$")
    _known_fields(
        root,
        {
            "schema_version",
            "id",
            "contract_version",
            "description",
            "model_requirements",
            "compile",
            "verify",
            "holdout",
            "statistics",
            "generation",
        },
        "$",
    )
    requirements_obj = _mapping(root.get("model_requirements", {}), "$.model_requirements")
    _known_fields(
        requirements_obj,
        {
            "tokenizer_hash",
            "base_signature",
            "architecture_hash",
            "state_schema_hash",
            "adapter_id",
            "output_semantics",
        },
        "$.model_requirements",
    )
    compile_obj = _mapping(root.get("compile", {}), "$.compile")
    _known_fields(compile_obj, {"objectives"}, "$.compile")
    objective_values = _list(
        compile_obj.get("objectives", []),
        "$.compile.objectives",
        maximum=limits.max_objectives,
    )
    verify_obj = _mapping(root.get("verify", {}), "$.verify")
    _known_fields(verify_obj, {"targets", "guards"}, "$.verify")
    target_values = _list(
        verify_obj.get("targets", []), "$.verify.targets", maximum=limits.max_assertions
    )
    guard_values = _list(
        verify_obj.get("guards", []), "$.verify.guards", maximum=limits.max_assertions
    )
    if len(target_values) + len(guard_values) > limits.max_assertions:
        raise ContractResourceLimitError(
            f"verification assertions exceed {limits.max_assertions} entries"
        )
    holdout_obj = _mapping(root.get("holdout", {}), "$.holdout")
    _known_fields(holdout_obj, {"sealed", "targets", "guards", "unseal_policy"}, "$.holdout")
    statistics_obj = _mapping(root.get("statistics", {}), "$.statistics")
    _known_fields(
        statistics_obj,
        {"confidence_level", "bootstrap_samples", "bootstrap_seed", "multiple_comparison"},
        "$.statistics",
    )
    generation_obj = _mapping(root.get("generation", {}), "$.generation")
    _known_fields(
        generation_obj,
        {
            "mode",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "seeds",
            "stop_sequences",
        },
        "$.generation",
    )
    try:
        model_requirements = ModelRequirements(
            tokenizer_hash=_string(
                requirements_obj.get("tokenizer_hash"),
                "$.model_requirements.tokenizer_hash",
                required=False,
            ),
            base_signature=_string(
                requirements_obj.get("base_signature"),
                "$.model_requirements.base_signature",
                required=False,
            ),
            architecture_hash=_string(
                requirements_obj.get("architecture_hash"),
                "$.model_requirements.architecture_hash",
                required=False,
            ),
            state_schema_hash=_string(
                requirements_obj.get("state_schema_hash"),
                "$.model_requirements.state_schema_hash",
                required=False,
            ),
            adapter_id=_string(
                requirements_obj.get("adapter_id"),
                "$.model_requirements.adapter_id",
                required=False,
            ),
            output_semantics=_required_string(
                requirements_obj.get("output_semantics", "causal_lm"),
                "$.model_requirements.output_semantics",
            ),
        )
        objectives = tuple(
            _parse_objective(item, f"$.compile.objectives[{index}]")
            for index, item in enumerate(objective_values)
        )
        targets = tuple(
            _parse_assertion(item, f"$.verify.targets[{index}]")
            for index, item in enumerate(target_values)
        )
        guards = tuple(
            _parse_assertion(item, f"$.verify.guards[{index}]")
            for index, item in enumerate(guard_values)
        )
        sealed = _boolean(holdout_obj.get("sealed", True), "$.holdout.sealed")
        default_unseal = (
            UnsealPolicy.FINAL_CANDIDATE_ONLY.value
            if sealed
            else UnsealPolicy.INDEPENDENT_VERIFICATION.value
        )
        holdout = HoldoutPolicy(
            sealed=sealed,
            targets=_string(holdout_obj.get("targets"), "$.holdout.targets", required=False),
            guards=_string(holdout_obj.get("guards"), "$.holdout.guards", required=False),
            unseal_policy=UnsealPolicy(
                _required_string(
                    holdout_obj.get("unseal_policy", default_unseal),
                    "$.holdout.unseal_policy",
                )
            ),
        )
        statistics = StatisticsPolicy(
            confidence_level=_number(
                statistics_obj.get("confidence_level", 0.95), "$.statistics.confidence_level"
            ),
            bootstrap_samples=_integer(
                statistics_obj.get("bootstrap_samples", 2000),
                "$.statistics.bootstrap_samples",
            ),
            bootstrap_seed=_integer(
                statistics_obj.get("bootstrap_seed", 81273), "$.statistics.bootstrap_seed"
            ),
            multiple_comparison=_required_string(
                statistics_obj.get("multiple_comparison", "none"),
                "$.statistics.multiple_comparison",
            ),
        )
        mode = GenerationMode(
            _required_string(generation_obj.get("mode", "greedy"), "$.generation.mode")
        )
        seed_values = _list(generation_obj.get("seeds", [0]), "$.generation.seeds", maximum=1024)
        stop_values = _list(
            generation_obj.get("stop_sequences", []),
            "$.generation.stop_sequences",
            maximum=128,
        )
        generation = GenerationPolicy(
            mode=mode,
            max_new_tokens=_integer(
                generation_obj.get("max_new_tokens", 128), "$.generation.max_new_tokens"
            ),
            temperature=_number(generation_obj.get("temperature", 1.0), "$.generation.temperature"),
            top_k=(
                None
                if generation_obj.get("top_k") is None
                else _integer(generation_obj["top_k"], "$.generation.top_k")
            ),
            top_p=_number(generation_obj.get("top_p", 1.0), "$.generation.top_p"),
            seeds=tuple(
                _integer(item, f"$.generation.seeds[{index}]")
                for index, item in enumerate(seed_values)
            ),
            stop_sequences=tuple(
                _required_string(item, f"$.generation.stop_sequences[{index}]")
                for index, item in enumerate(stop_values)
            ),
        )
        return BehaviorContract(
            schema_version=_integer(root.get("schema_version"), "$.schema_version"),
            id=_required_string(root.get("id"), "$.id"),
            contract_version=_integer(root.get("contract_version", 1), "$.contract_version"),
            description=_string(root.get("description"), "$.description", required=False),
            model_requirements=model_requirements,
            objectives=objectives,
            targets=targets,
            guards=guards,
            holdout=holdout,
            statistics=statistics,
            generation=generation,
        )
    except ContractError:
        raise
    except (TypeError, ValueError) as error:
        _fail("$", str(error))


def loads_contract(
    text: str | bytes,
    *,
    format: str = "yaml",
    limits: ContractLimits = DEFAULT_LIMITS,
) -> BehaviorContract:
    return parse_contract(loads_data(text, format=format, limits=limits), limits=limits)


def load_contract(
    path: str | Path,
    *,
    limits: ContractLimits = DEFAULT_LIMITS,
) -> BehaviorContract:
    return parse_contract(load_data_file(path, limits=limits), limits=limits)


def canonical_contract_json(contract: BehaviorContract) -> str:
    return canonical_dumps(contract.to_dict(), max_depth=DEFAULT_LIMITS.max_depth)


def resolve_contract_resource(contract_path: str | Path, resource: str) -> Path:
    """Resolve an untrusted contract-relative resource without directory escape."""

    return resolve_inside(Path(contract_path).resolve().parent, resource)


__all__ = [
    "DEFAULT_LIMITS",
    "ContractError",
    "ContractLimits",
    "ContractResourceLimitError",
    "ContractSyntaxError",
    "ContractValidationError",
    "canonical_contract_json",
    "load_contract",
    "load_data_file",
    "loads_contract",
    "loads_data",
    "parse_contract",
    "resolve_contract_resource",
    "validate_data_shape",
]
