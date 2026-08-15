"""Bounded multi-contract CEGIS for semantic merge and rebase.

The generic compiler CEGIS entry point operates on one contract.  Semantic
merge and rebase instead have to search a union of independently packaged
contracts while retaining their distinct resources and identities.  This
module provides that union search without weakening unsupported assertions or
opening sealed holdouts.
"""

from __future__ import annotations

import dataclasses
import math
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from modelpact.adapters.base import GenerationPolicy, ModelAdapter, ModelBatch
from modelpact.checkpoints.safetensors import tensor_content_hash
from modelpact.compiler.cegis import CEGISResult, Counterexample, run_cegis
from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
from modelpact.compiler.result import CompilationResult
from modelpact.contracts.ast import AssertionType, BehaviorContract, GenerationMode
from modelpact.probes.minimize import minimize_prompt
from modelpact.probes.mutations import mutate_prompt
from modelpact.util.hashing import hash_canonical, sha256_bytes
from modelpact.verify.provider import load_probe_records


class SemanticCEGISUnsupportedError(ValueError):
    """Raised when the declared union has no honest executable search semantics."""


@dataclass(frozen=True, slots=True)
class ScopedSearchExample:
    contract_id: str
    domain: str
    assertion_id: str
    assertion_type: str
    record_id: str
    source_prompt: str
    prompt: str
    expected: str | None
    maximum_kl: float | None
    mutation_operator: str

    def __post_init__(self) -> None:
        if self.domain not in {"target", "guard"}:
            raise ValueError("semantic search domain must be target or guard")
        if not self.contract_id or not self.assertion_id or not self.record_id:
            raise ValueError("semantic search examples require stable identities")
        if (
            self.domain == "target"
            and self.assertion_type != AssertionType.GENERATION_LENGTH.value
            and self.expected is None
        ):
            raise ValueError("target search examples require an expected completion")
        if self.domain == "guard" and self.maximum_kl is None:
            raise ValueError("guard search examples require a KL threshold")


@dataclass(frozen=True, slots=True)
class SearchObservation:
    example: ScopedSearchExample
    passed: bool | None
    margin: float | None
    output_hash: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticSearchExecution:
    round_index: int
    domain: str
    proposals: tuple[ScopedSearchExample, ...]
    observations: tuple[SearchObservation, ...]
    model_executions: int
    minimization_executions: int
    invalid_candidates: int
    candidate_space_size: int
    candidate_space_truncated: bool


@dataclass(frozen=True, slots=True)
class SemanticCompilationExecution:
    candidate_index: int
    target_example_ids: tuple[str, ...]
    guard_example_ids: tuple[str, ...]
    candidate_id: str | None
    feasible: bool
    optimization_steps: int
    violated_constraints: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SemanticCEGISRun:
    result: CEGISResult[ScopedSearchExample]
    target_candidates: tuple[ScopedSearchExample, ...]
    guard_candidates: tuple[ScopedSearchExample, ...]
    search_executions: tuple[SemanticSearchExecution, ...]
    compilation_executions: tuple[SemanticCompilationExecution, ...]
    unsupported_search_assertions: Mapping[str, str]
    maximum_rounds: int
    search_budget_per_domain_per_round: int
    candidate_space_truncated: bool

    @property
    def candidate(self) -> CompilationResult:
        return self.result.candidate

    def to_dict(self) -> dict[str, object]:
        def example(
            item: ScopedSearchExample,
            *,
            include_expected: bool = False,
        ) -> dict[str, object]:
            value: dict[str, object] = {
                "assertion_id": item.assertion_id,
                "assertion_type": item.assertion_type,
                "contract_id": item.contract_id,
                "domain": item.domain,
                "maximum_kl": item.maximum_kl,
                "mutation_operator": item.mutation_operator,
                "prompt": item.prompt,
                "prompt_hash": hash_canonical({"prompt": item.prompt}),
                "record_id": item.record_id,
                "source_prompt_hash": hash_canonical({"prompt": item.source_prompt}),
            }
            if include_expected and item.expected is not None:
                value["expected"] = item.expected
                value["expected_hash"] = sha256_bytes(item.expected.encode("utf-8"))
            return value

        def identifier(item: ScopedSearchExample) -> str:
            return hash_canonical(example(item, include_expected=True))

        return {
            "schema_version": 1,
            "outcome": self.result.stop_reason.value,
            "stop_reason": self.result.stop_reason.value,
            "maximum_rounds": self.maximum_rounds,
            "rounds_executed": len(self.result.rounds),
            "search_budget_per_domain_per_round": self.search_budget_per_domain_per_round,
            "candidate_space_truncated": self.candidate_space_truncated,
            "target_candidate_space": len(self.target_candidates),
            "guard_candidate_space": len(self.guard_candidates),
            "model_executions": sum(item.model_executions for item in self.search_executions),
            "minimization_executions": sum(
                item.minimization_executions for item in self.search_executions
            ),
            "unsupported_search_assertions": dict(
                sorted(self.unsupported_search_assertions.items())
            ),
            "compilation_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "candidate_index": item.candidate_index,
                    "feasible": item.feasible,
                    "guard_example_ids": list(item.guard_example_ids),
                    "optimization_steps": item.optimization_steps,
                    "target_example_ids": list(item.target_example_ids),
                    "violated_constraints": dict(sorted(item.violated_constraints.items())),
                }
                for item in self.compilation_executions
            ],
            "search_executions": [
                {
                    "candidate_space_size": item.candidate_space_size,
                    "candidate_space_truncated": item.candidate_space_truncated,
                    "domain": item.domain,
                    "invalid_candidates": item.invalid_candidates,
                    "minimization_executions": item.minimization_executions,
                    "model_executions": item.model_executions,
                    "observations": [
                        {
                            **example(observation.example),
                            "margin": observation.margin,
                            "output_hash": observation.output_hash,
                            "passed": observation.passed,
                            "status": (
                                "INVALID"
                                if observation.passed is None
                                else ("PASS" if observation.passed else "FAIL")
                            ),
                            "error": observation.error,
                        }
                        for observation in item.observations
                    ],
                    "proposals": [
                        example(proposal, include_expected=proposal.domain == "target")
                        for proposal in item.proposals
                    ],
                    "round_index": item.round_index,
                }
                for item in self.search_executions
            ],
            "working_target_examples": [
                example(item, include_expected=True) for item in self.result.working_target_examples
            ],
            "working_guard_examples": [
                example(item) for item in self.result.working_guard_examples
            ],
            "rounds": [
                {
                    "compilation_feasible": item.compilation_feasible,
                    "guard_counterexamples": [
                        {
                            **example(counterexample.example),
                            "example_id": identifier(counterexample.example),
                            "margin": counterexample.margin,
                            "minimized": counterexample.minimized,
                            "provenance": dict(sorted(counterexample.provenance.items())),
                        }
                        for counterexample in item.guard_counterexamples
                    ],
                    "round_index": item.round_index,
                    "search_budget_per_domain": item.search_budget,
                    "target_counterexamples": [
                        {
                            **example(counterexample.example, include_expected=True),
                            "example_id": identifier(counterexample.example),
                            "margin": counterexample.margin,
                            "minimized": counterexample.minimized,
                            "provenance": dict(sorted(counterexample.provenance.items())),
                        }
                        for counterexample in item.target_counterexamples
                    ],
                }
                for item in self.result.rounds
            ],
            "scope": (
                "deterministic mutations of every supported visible assertion in the "
                "declared contract union; sealed holdouts are excluded"
            ),
        }


@dataclass(frozen=True, slots=True)
class _TargetBatch:
    batch: ModelBatch
    labels: Tensor
    token_mask: Tensor


@dataclass(frozen=True, slots=True)
class _GuardBatch:
    batch: ModelBatch
    base_logits: Tensor


@dataclass(slots=True)
class _Counters:
    callback_calls: dict[str, int] = field(default_factory=lambda: {"target": 0, "guard": 0})
    searches: list[SemanticSearchExecution] = field(default_factory=list)
    compilations: list[SemanticCompilationExecution] = field(default_factory=list)


ContractSource = tuple[BehaviorContract, Path]
CandidateCompiler = Callable[
    [tuple[ScopedSearchExample, ...], tuple[ScopedSearchExample, ...]], CompilationResult
]
CandidateModel = Callable[[Mapping[str, Tensor]], nn.Module]


def _maximum_kl(options: Mapping[str, object]) -> float:
    values: list[float] = []
    for key in ("maximum", "maximum_mean", "maximum_item"):
        value = options.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    quantile = options.get("maximum_quantile")
    if isinstance(quantile, Mapping):
        value = quantile.get("value")
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise SemanticCEGISUnsupportedError("base-KL search requires a finite non-negative maximum")
    return min(values)


def _expected(record: Mapping[str, object], options: Mapping[str, object]) -> str | None:
    value = record.get("expected", options.get("expected"))
    return value if isinstance(value, str) else None


def _supported_target(assertion_type: AssertionType, options: Mapping[str, object]) -> str | None:
    if assertion_type is AssertionType.GENERATION_LENGTH:
        unit = options.get("unit", "tokens")
        if unit not in {"tokens", "characters", "words"}:
            return f"unsupported generation_length unit: {unit!r}"
        return None
    if assertion_type is AssertionType.EXACT_MATCH:
        return None
    if assertion_type is not AssertionType.FREE_GENERATION_MATCH:
        return (
            "search supports only exact_match, free_generation_match, and generation_length targets"
        )
    match_type = options.get("match_type", "exact")
    if match_type not in {"exact", "normalized", "contains"}:
        return f"unsupported free_generation_match search mode: {match_type!r}"
    return None


def _build_union_plan(
    contracts: Mapping[str, ContractSource],
    *,
    maximum_rounds: int,
    search_budget: int,
    seed: int,
) -> tuple[
    tuple[ScopedSearchExample, ...],
    tuple[ScopedSearchExample, ...],
    tuple[ScopedSearchExample, ...],
    tuple[ScopedSearchExample, ...],
    dict[str, str],
    bool,
]:
    maximum_candidates = maximum_rounds * search_budget
    maximum_seeds = max(1, min(256, maximum_candidates))
    target_seeds: list[ScopedSearchExample] = []
    guard_seeds: list[ScopedSearchExample] = []
    target_candidate_groups: list[list[ScopedSearchExample]] = []
    guard_candidate_groups: list[list[ScopedSearchExample]] = []
    unsupported: dict[str, str] = {}
    target_seen: set[tuple[str, str, str, str | None]] = set()
    guard_seen: set[tuple[str, str, str, float | None]] = set()
    truncated = False

    def mutations(item: ScopedSearchExample, *, offset: int) -> tuple[ScopedSearchExample, ...]:
        return tuple(
            dataclasses.replace(
                item,
                prompt=mutation.mutated,
                mutation_operator=mutation.operator.value,
            )
            for mutation in mutate_prompt(item.prompt, seed=seed + offset)
        )

    for contract_index, (contract_id, (contract, contract_path)) in enumerate(
        sorted(contracts.items())
    ):
        if contract.generation.mode is not GenerationMode.GREEDY:
            for assertion in contract.targets:
                unsupported[f"{contract_id}:{assertion.id}"] = (
                    "semantic CEGIS requires deterministic greedy generation"
                )
            continue
        root = contract_path.resolve().parent
        for assertion_index, assertion in enumerate(contract.targets):
            namespaced = f"{contract_id}:{assertion.id}"
            reason = _supported_target(assertion.type, assertion.options)
            if reason is not None:
                unsupported[namespaced] = reason
                continue
            records = load_probe_records(root, assertion.source)
            valid_assertion = True
            local_seeds: list[ScopedSearchExample] = []
            local_candidates: list[ScopedSearchExample] = []
            for record_index, record in enumerate(records):
                expected = _expected(record, assertion.options)
                if expected is None and assertion.type is not AssertionType.GENERATION_LENGTH:
                    unsupported[namespaced] = "target search records require an expected string"
                    valid_assertion = False
                    break
                prompt = str(record["prompt"])
                item = ScopedSearchExample(
                    contract_id,
                    "target",
                    assertion.id,
                    assertion.type.value,
                    str(record["id"]),
                    prompt,
                    prompt,
                    expected,
                    None,
                    "seed",
                )
                local_seeds.append(item)
                for candidate in mutations(
                    item,
                    offset=(
                        contract_index * 10_000_019 + assertion_index * 1_000_003 + record_index
                    ),
                ):
                    key = (
                        candidate.contract_id,
                        candidate.assertion_id,
                        candidate.prompt,
                        candidate.expected,
                    )
                    if key in target_seen:
                        continue
                    target_seen.add(key)
                    if len(local_candidates) < maximum_candidates:
                        local_candidates.append(candidate)
                    else:
                        truncated = True
                        break
            if not valid_assertion:
                continue
            for item in local_seeds:
                if len(target_seeds) < maximum_seeds:
                    target_seeds.append(item)
                else:
                    truncated = True
            target_candidate_groups.append(local_candidates[:maximum_candidates])
            if len(local_candidates) > maximum_candidates:
                truncated = True

        for assertion_index, assertion in enumerate(contract.guards):
            namespaced = f"{contract_id}:{assertion.id}"
            if assertion.type is not AssertionType.BASE_KL:
                unsupported[namespaced] = "guard search supports only base_kl assertions"
                continue
            try:
                maximum = _maximum_kl(assertion.options)
            except SemanticCEGISUnsupportedError as error:
                unsupported[namespaced] = str(error)
                continue
            records = load_probe_records(root, assertion.source)
            local_candidates = []
            for record_index, record in enumerate(records):
                prompt = str(record["prompt"])
                item = ScopedSearchExample(
                    contract_id,
                    "guard",
                    assertion.id,
                    assertion.type.value,
                    str(record["id"]),
                    prompt,
                    prompt,
                    None,
                    maximum,
                    "seed",
                )
                if len(guard_seeds) < maximum_seeds:
                    guard_seeds.append(item)
                else:
                    truncated = True
                for candidate in mutations(
                    item,
                    offset=(
                        5_000_021
                        + contract_index * 10_000_019
                        + assertion_index * 1_000_003
                        + record_index
                    ),
                ):
                    guard_key = (
                        candidate.contract_id,
                        candidate.assertion_id,
                        candidate.prompt,
                        candidate.maximum_kl,
                    )
                    if guard_key in guard_seen:
                        continue
                    guard_seen.add(guard_key)
                    if len(local_candidates) < maximum_candidates:
                        local_candidates.append(candidate)
                    else:
                        truncated = True
                        break
            guard_candidate_groups.append(local_candidates)

    def balanced(
        groups: list[list[ScopedSearchExample]],
    ) -> tuple[ScopedSearchExample, ...]:
        nonlocal truncated
        selected: list[ScopedSearchExample] = []
        position = 0
        while len(selected) < maximum_candidates:
            added = False
            for group in groups:
                if position < len(group):
                    selected.append(group[position])
                    added = True
                    if len(selected) >= maximum_candidates:
                        break
            if not added:
                break
            position += 1
        if sum(len(group) for group in groups) > len(selected):
            truncated = True
        return tuple(selected)

    target_candidates = balanced(target_candidate_groups)
    guard_candidates = balanced(guard_candidate_groups)

    if not target_seeds:
        detail = "; ".join(
            f"{identifier}: {reason}" for identifier, reason in sorted(unsupported.items())
        )
        raise SemanticCEGISUnsupportedError(
            "semantic CEGIS requires at least one supported generative target with "
            f"visible probes{f' ({detail})' if detail else ''}"
        )
    if not guard_seeds:
        detail = "; ".join(
            f"{identifier}: {reason}" for identifier, reason in sorted(unsupported.items())
        )
        raise SemanticCEGISUnsupportedError(
            "semantic CEGIS requires at least one visible base_kl preservation guard"
            f"{f' ({detail})' if detail else ''}"
        )
    return (
        tuple(target_seeds),
        tuple(guard_seeds),
        target_candidates,
        guard_candidates,
        unsupported,
        truncated,
    )


def _target_batch(adapter: ModelAdapter, example: ScopedSearchExample) -> _TargetBatch:
    assert example.expected is not None
    tokenizer = adapter.tokenizer()
    prompt_ids = tokenizer.encode(example.prompt, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(example.expected, add_bos=False, add_eos=True)
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    token_mask = torch.zeros_like(attention)
    token_mask[:, len(prompt_ids) :] = True
    return _TargetBatch(ModelBatch(input_ids, attention), input_ids.clone(), token_mask)


def _target_loss(adapter: ModelAdapter) -> object:
    def loss(model: nn.Module, example: _TargetBatch) -> Tensor:
        logits = adapter.forward_logits(model, example.batch)
        shifted_logits = logits[:, :-1].contiguous()
        labels = example.labels[:, 1:].to(device=logits.device)
        mask = example.token_mask[:, 1:].to(device=logits.device)
        per_token = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        selected = per_token.masked_select(mask)
        if selected.numel() == 0:
            raise ValueError("semantic CEGIS target completion produced no trainable tokens")
        return selected.mean()

    return loss


def _guard_batch(
    adapter: ModelAdapter,
    base_model: nn.Module,
    example: ScopedSearchExample,
) -> _GuardBatch:
    batch = adapter.tokenizer().batch((example.prompt,), add_bos=True)
    with torch.no_grad():
        logits = adapter.forward_logits(base_model, batch).detach().cpu()
    return _GuardBatch(batch, logits)


def _guard_kl(adapter: ModelAdapter) -> object:
    def measure(model: nn.Module, example: _GuardBatch) -> Tensor:
        student = adapter.forward_logits(model, example.batch).to(torch.float64)
        reference = example.base_logits.to(device=student.device, dtype=student.dtype)
        reference_log = torch.log_softmax(reference, dim=-1)
        student_log = torch.log_softmax(student, dim=-1)
        per_token = (reference_log.exp() * (reference_log - student_log)).sum(dim=-1)
        mask = example.batch.attention_mask.to(device=student.device, dtype=student.dtype)
        return (per_token * mask).sum() / mask.sum().clamp_min(1.0)

    return measure


def differentiable_refinement_problem(
    adapter: ModelAdapter,
    base_model: nn.Module,
    target_examples: tuple[ScopedSearchExample, ...],
    guard_examples: tuple[ScopedSearchExample, ...],
) -> tuple[tuple[DifferentiableObjective, ...], tuple[DifferentiableConstraint, ...]]:
    """Build differentiable CE and base-KL terms for accumulated examples."""

    objectives: list[DifferentiableObjective] = []
    differentiable_targets = tuple(item for item in target_examples if item.expected is not None)
    if differentiable_targets:
        objectives.append(
            DifferentiableObjective(
                "semantic-cegis-target-cross-entropy",
                tuple(_target_batch(adapter, item) for item in differentiable_targets),
                _target_loss(adapter),  # type: ignore[arg-type]
            )
        )
    grouped: dict[tuple[str, str, float], list[ScopedSearchExample]] = {}
    for item in guard_examples:
        assert item.maximum_kl is not None
        grouped.setdefault((item.contract_id, item.assertion_id, item.maximum_kl), []).append(item)
    constraints = tuple(
        DifferentiableConstraint(
            f"semantic-cegis-guard-{index:04d}",
            tuple(_guard_batch(adapter, base_model, item) for item in examples),
            _guard_kl(adapter),  # type: ignore[arg-type]
            maximum,
        )
        for index, ((_contract_id, _assertion_id, maximum), examples) in enumerate(
            sorted(grouped.items())
        )
    )
    return tuple(objectives), constraints


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _target_passes(
    adapter: ModelAdapter,
    model: nn.Module,
    contracts: Mapping[str, ContractSource],
    example: ScopedSearchExample,
) -> tuple[bool, str]:
    contract = contracts[example.contract_id][0]
    assertion = next(item for item in contract.targets if item.id == example.assertion_id)
    policy = GenerationPolicy(
        mode="greedy",
        max_new_tokens=contract.generation.max_new_tokens,
        seed=contract.generation.seeds[0],
        temperature=contract.generation.temperature,
    )
    sample = adapter.generate(model, adapter.tokenizer().batch((example.prompt,)), policy)[0]
    if assertion.type is AssertionType.GENERATION_LENGTH:
        minimum = assertion.options.get("minimum", 0)
        maximum = assertion.options.get("maximum", contract.generation.max_new_tokens)
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
        ):
            raise SemanticCEGISUnsupportedError(
                "generation_length search requires integer minimum and maximum"
            )
        unit = assertion.options.get("unit", "tokens")
        if unit == "tokens":
            length = len(sample.token_ids)
        elif unit == "characters":
            length = len(sample.text)
        elif unit == "words":
            length = len(sample.text.split())
        else:
            raise SemanticCEGISUnsupportedError(f"unsupported generation_length unit: {unit!r}")
        return minimum <= length <= maximum, sample.text
    expected = example.expected
    assert expected is not None
    case_sensitive = assertion.options.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        raise SemanticCEGISUnsupportedError("case_sensitive must be boolean")
    actual_cmp = sample.text if case_sensitive else sample.text.casefold()
    expected_cmp = expected if case_sensitive else expected.casefold()
    if assertion.type is AssertionType.EXACT_MATCH:
        return actual_cmp == expected_cmp, sample.text
    match_type = assertion.options.get("match_type", "exact")
    if match_type == "exact":
        passed = actual_cmp == expected_cmp
    elif match_type == "normalized":
        actual_normalized = _normalize(sample.text)
        expected_normalized = _normalize(expected)
        if not case_sensitive:
            actual_normalized = actual_normalized.casefold()
            expected_normalized = expected_normalized.casefold()
        passed = actual_normalized == expected_normalized
    elif match_type == "contains":
        passed = expected_cmp in actual_cmp
    else:
        raise SemanticCEGISUnsupportedError(
            f"unsupported free_generation_match search mode: {match_type!r}"
        )
    return passed, sample.text


def _guard_value(
    adapter: ModelAdapter,
    base_model: nn.Module,
    model: nn.Module,
    prompt: str,
) -> float:
    batch = adapter.tokenizer().batch((prompt,), add_bos=True)
    with torch.no_grad():
        base_logits = adapter.forward_logits(base_model, batch).to(torch.float64)
        student_logits = adapter.forward_logits(model, batch).to(torch.float64)
    base_log = torch.log_softmax(base_logits, dim=-1)
    student_log = torch.log_softmax(student_logits, dim=-1)
    per_token = (base_log.exp() * (base_log - student_log)).sum(dim=-1)
    mask = batch.attention_mask.to(device=per_token.device, dtype=per_token.dtype)
    return float(((per_token * mask).sum() / mask.sum().clamp_min(1.0)).detach().cpu())


def _candidate_id(candidate: CompilationResult) -> str | None:
    if not candidate.deltas:
        return None
    return hash_canonical(
        {
            name: tensor_content_hash(value.detach().cpu())
            for name, value in sorted(candidate.deltas.items())
        }
    )


def _example_id(item: ScopedSearchExample) -> str:
    return hash_canonical(
        {
            "assertion_id": item.assertion_id,
            "assertion_type": item.assertion_type,
            "contract_id": item.contract_id,
            "domain": item.domain,
            "expected": item.expected,
            "maximum_kl": item.maximum_kl,
            "mutation_operator": item.mutation_operator,
            "prompt": item.prompt,
            "record_id": item.record_id,
            "source_prompt": item.source_prompt,
        }
    )


def run_semantic_cegis(
    adapter: ModelAdapter,
    base_model: nn.Module,
    contracts: Mapping[str, ContractSource],
    *,
    compile_candidate: CandidateCompiler,
    candidate_model: CandidateModel,
    maximum_rounds: int,
    search_budget_per_domain_per_round: int,
    seed: int,
) -> SemanticCEGISRun:
    """Compile, execute union mutations, reduce failures, and recompile."""

    if maximum_rounds <= 0 or search_budget_per_domain_per_round <= 0:
        raise ValueError("semantic CEGIS rounds and search budget must be positive")
    if maximum_rounds * search_budget_per_domain_per_round > 100_000:
        raise SemanticCEGISUnsupportedError(
            "semantic CEGIS supports at most 100000 proposed mutations per domain"
        )
    (
        target_seeds,
        guard_seeds,
        target_candidates,
        guard_candidates,
        unsupported,
        truncated,
    ) = _build_union_plan(
        contracts,
        maximum_rounds=maximum_rounds,
        search_budget=search_budget_per_domain_per_round,
        seed=seed,
    )
    counters = _Counters()

    def recorded_compile(
        targets: tuple[ScopedSearchExample, ...],
        guards: tuple[ScopedSearchExample, ...],
    ) -> CompilationResult:
        candidate = compile_candidate(targets, guards)
        optimization_steps = candidate.metadata.get("optimization_steps")
        if not isinstance(optimization_steps, int) or isinstance(optimization_steps, bool):
            optimization_steps = len(candidate.evidence)
        counters.compilations.append(
            SemanticCompilationExecution(
                len(counters.compilations),
                tuple(_example_id(item) for item in targets),
                tuple(_example_id(item) for item in guards),
                _candidate_id(candidate),
                candidate.feasible,
                optimization_steps,
                dict(candidate.violated_constraints),
            )
        )
        return candidate

    def search_targets(
        candidate: CompilationResult,
        budget: int,
    ) -> tuple[Counterexample[ScopedSearchExample], ...]:
        round_index = counters.callback_calls["target"]
        counters.callback_calls["target"] += 1
        start = round_index * budget
        proposals = target_candidates[start : start + budget]
        model = candidate_model(candidate.deltas)
        executions = 0
        minimization_executions = 0
        invalid = 0
        observations: list[SearchObservation] = []
        found: list[Counterexample[ScopedSearchExample]] = []
        for item in proposals:
            executions += 1
            try:
                # Validate the prompt against the declared model input domain.
                # Tiny generation stops without error at its context boundary,
                # which is not a behavioral counterexample to generation length.
                validation_batch = adapter.tokenizer().batch((item.prompt,))
                with torch.no_grad():
                    adapter.forward_logits(base_model, validation_batch)
                executions += 1
                passed, output = _target_passes(adapter, model, contracts, item)
            except ValueError as error:
                invalid += 1
                observations.append(SearchObservation(item, None, None, error=type(error).__name__))
                continue
            observations.append(
                SearchObservation(
                    item,
                    passed,
                    1.0 if passed else -1.0,
                    sha256_bytes(output.encode("utf-8")),
                )
            )
            if passed:
                continue

            def preserves_failure(prompt: str, *, current: ScopedSearchExample = item) -> bool:
                nonlocal executions, minimization_executions
                executions += 1
                minimization_executions += 1
                return not _target_passes(
                    adapter,
                    model,
                    contracts,
                    dataclasses.replace(current, prompt=prompt),
                )[0]

            minimized = minimize_prompt(item.prompt, preserves_failure)
            reduced = dataclasses.replace(item, prompt=minimized.minimized)
            retained = preserves_failure(reduced.prompt)
            if not retained:
                reduced = item
            found.append(
                Counterexample(
                    reduced,
                    "target",
                    -1.0,
                    retained and reduced.prompt != item.prompt,
                    {
                        "accepted_reductions": minimized.accepted_reductions,
                        "generated_output_hash": sha256_bytes(output.encode("utf-8")),
                        "minimization_evaluations": minimized.evaluations,
                        "minimization_preserved_failure": retained,
                        "original_prompt_hash": hash_canonical({"prompt": item.prompt}),
                    },
                )
            )
        counters.searches.append(
            SemanticSearchExecution(
                round_index,
                "target",
                proposals,
                tuple(observations),
                executions,
                minimization_executions,
                invalid,
                len(target_candidates),
                truncated,
            )
        )
        return tuple(found)

    def search_guards(
        candidate: CompilationResult,
        budget: int,
    ) -> tuple[Counterexample[ScopedSearchExample], ...]:
        round_index = counters.callback_calls["guard"]
        counters.callback_calls["guard"] += 1
        start = round_index * budget
        proposals = guard_candidates[start : start + budget]
        model = candidate_model(candidate.deltas)
        executions = 0
        minimization_executions = 0
        invalid = 0
        observations: list[SearchObservation] = []
        found: list[Counterexample[ScopedSearchExample]] = []
        for item in proposals:
            maximum = item.maximum_kl
            assert maximum is not None
            maximum_value = maximum
            executions += 1
            try:
                value = _guard_value(adapter, base_model, model, item.prompt)
            except ValueError as error:
                invalid += 1
                observations.append(SearchObservation(item, None, None, error=type(error).__name__))
                continue
            margin = maximum_value - value
            observations.append(SearchObservation(item, margin >= 0.0, margin))
            if margin >= 0.0:
                continue

            def preserves_failure(
                prompt: str,
                *,
                threshold: float = maximum_value,
            ) -> bool:
                nonlocal executions, minimization_executions
                executions += 1
                minimization_executions += 1
                return _guard_value(adapter, base_model, model, prompt) > threshold

            minimized = minimize_prompt(item.prompt, preserves_failure)
            reduced = dataclasses.replace(item, prompt=minimized.minimized)
            minimized_value = _guard_value(adapter, base_model, model, reduced.prompt)
            executions += 1
            minimization_executions += 1
            retained = minimized_value > maximum_value
            if not retained:
                reduced = item
                minimized_value = value
            found.append(
                Counterexample(
                    reduced,
                    "guard",
                    maximum_value - minimized_value,
                    retained and reduced.prompt != item.prompt,
                    {
                        "accepted_reductions": minimized.accepted_reductions,
                        "maximum_kl": maximum_value,
                        "minimization_evaluations": minimized.evaluations,
                        "minimization_preserved_failure": retained,
                        "original_prompt_hash": hash_canonical({"prompt": item.prompt}),
                        "retained_observed_kl": minimized_value,
                    },
                )
            )
        counters.searches.append(
            SemanticSearchExecution(
                round_index,
                "guard",
                proposals,
                tuple(observations),
                executions,
                minimization_executions,
                invalid,
                len(guard_candidates),
                truncated,
            )
        )
        return tuple(found)

    result = run_cegis(
        target_seeds,
        guard_seeds,
        compile_candidate=recorded_compile,
        search_targets=search_targets,
        search_guards=search_guards,
        maximum_rounds=maximum_rounds,
        search_budget_per_round=search_budget_per_domain_per_round,
    )
    return SemanticCEGISRun(
        result,
        target_candidates,
        guard_candidates,
        tuple(counters.searches),
        tuple(counters.compilations),
        unsupported,
        maximum_rounds,
        search_budget_per_domain_per_round,
        truncated,
    )


def candidate_satisfies_semantic_working_set(
    run: SemanticCEGISRun,
    adapter: ModelAdapter,
    base_model: nn.Module,
    contracts: Mapping[str, ContractSource],
    candidate_model: CandidateModel,
    deltas: Mapping[str, Tensor],
) -> bool:
    """Execute the accumulated visible search set against a candidate."""

    model = candidate_model(deltas)
    for item in run.result.working_target_examples:
        if not _target_passes(adapter, model, contracts, item)[0]:
            return False
    for item in run.result.working_guard_examples:
        assert item.maximum_kl is not None
        if _guard_value(adapter, base_model, model, item.prompt) > item.maximum_kl:
            return False
    return True


__all__ = [
    "ScopedSearchExample",
    "SemanticCEGISRun",
    "SemanticCEGISUnsupportedError",
    "candidate_satisfies_semantic_working_set",
    "differentiable_refinement_problem",
    "run_semantic_cegis",
]
