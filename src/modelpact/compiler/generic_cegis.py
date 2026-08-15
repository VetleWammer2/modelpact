"""Bounded deterministic CEGIS for executable generative contracts.

This module intentionally supports a narrow, auditable search semantics.  It
mutates non-holdout target and guard probe prompts, executes every selected
mutation, minimizes observed failures, and recompiles differentiable target CE
and base-KL examples.  Unsupported assertion semantics are reported instead of
being silently treated as searched.
"""

from __future__ import annotations

import dataclasses
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from modelpact.adapters.base import (
    GenerationPolicy as AdapterGenerationPolicy,
)
from modelpact.adapters.base import (
    ModelAdapter,
    ModelBatch,
)
from modelpact.compiler.cegis import CEGISResult, Counterexample, run_cegis
from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
from modelpact.compiler.contracts import PreparedContract
from modelpact.compiler.extract import apply_dense_deltas
from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
from modelpact.compiler.result import CompilationResult
from modelpact.contracts.ast import AssertionType, BehaviorContract, GenerationMode
from modelpact.probes.minimize import minimize_prompt
from modelpact.probes.mutations import mutate_prompt
from modelpact.util.hashing import hash_canonical, sha256_bytes
from modelpact.verify.provider import load_probe_records


class GenericCEGISUnsupportedError(ValueError):
    """Raised when a requested search has no honest executable semantics."""


@dataclass(frozen=True, slots=True)
class SearchExample:
    domain: str
    assertion_id: str
    record_id: str
    source_prompt: str
    prompt: str
    expected: str | None
    maximum_kl: float | None
    mutation_operator: str

    def __post_init__(self) -> None:
        if self.domain not in {"target", "guard"}:
            raise ValueError("search example domain must be target or guard")
        if not self.assertion_id or not self.record_id:
            raise ValueError("search examples require assertion and record identities")
        if self.domain == "target" and self.expected is None:
            raise ValueError("target search examples require an expected completion")
        if self.domain == "guard" and self.maximum_kl is None:
            raise ValueError("guard search examples require a KL threshold")


@dataclass(frozen=True, slots=True)
class SearchExecution:
    round_index: int
    domain: str
    proposed: int
    model_executions: int
    minimization_executions: int
    failures: int
    invalid_candidates: int
    candidate_space_size: int
    candidate_space_truncated: bool


@dataclass(frozen=True, slots=True)
class GenericCEGISRun:
    result: CEGISResult[SearchExample]
    target_seed_examples: tuple[SearchExample, ...]
    guard_seed_examples: tuple[SearchExample, ...]
    target_candidates: tuple[SearchExample, ...]
    guard_candidates: tuple[SearchExample, ...]
    executed_target_examples: tuple[SearchExample, ...]
    executed_guard_examples: tuple[SearchExample, ...]
    search_executions: tuple[SearchExecution, ...]
    unsupported_search_assertions: Mapping[str, str]
    maximum_rounds: int
    search_budget_per_domain_per_round: int
    candidate_space_truncated: bool

    @property
    def candidate(self) -> CompilationResult:
        return self.result.candidate

    def to_dict(self) -> dict[str, object]:
        def example(item: SearchExample, *, include_expected: bool = False) -> dict[str, object]:
            value: dict[str, object] = {
                "assertion_id": item.assertion_id,
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

        searches = [dataclasses.asdict(item) for item in self.search_executions]
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
            "search_executions": searches,
            "working_target_examples": [
                example(item, include_expected=True) for item in self.result.working_target_examples
            ],
            "working_guard_examples": [
                example(item) for item in self.result.working_guard_examples
            ],
            "executed_target_examples": [
                example(item, include_expected=True) for item in self.executed_target_examples
            ],
            "executed_guard_examples": [example(item) for item in self.executed_guard_examples],
            "rounds": [
                {
                    "round_index": item.round_index,
                    "search_budget_per_domain": item.search_budget,
                    "compilation_feasible": item.compilation_feasible,
                    "target_counterexamples": [
                        {
                            **example(counterexample.example, include_expected=True),
                            "margin": counterexample.margin,
                            "minimized": counterexample.minimized,
                            "provenance": dict(sorted(counterexample.provenance.items())),
                        }
                        for counterexample in item.target_counterexamples
                    ],
                    "guard_counterexamples": [
                        {
                            **example(counterexample.example),
                            "margin": counterexample.margin,
                            "minimized": counterexample.minimized,
                            "provenance": dict(sorted(counterexample.provenance.items())),
                        }
                        for counterexample in item.guard_counterexamples
                    ],
                }
                for item in self.result.rounds
            ],
            "scope": (
                "deterministic mutations of declared non-holdout target and guard probes; "
                "no unexecuted candidate is classified as passing"
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
class _SearchCounters:
    callback_calls: dict[str, int] = field(default_factory=lambda: {"target": 0, "guard": 0})
    records: list[SearchExecution] = field(default_factory=list)


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
        raise GenericCEGISUnsupportedError("base-KL search requires a finite non-negative maximum")
    return min(values)


def _expected(record: Mapping[str, object], options: Mapping[str, object]) -> str | None:
    value = record.get("expected", options.get("expected"))
    return value if isinstance(value, str) else None


def _supported_target(assertion_type: AssertionType, options: Mapping[str, object]) -> str | None:
    if assertion_type is AssertionType.EXACT_MATCH:
        return None
    if assertion_type is not AssertionType.FREE_GENERATION_MATCH:
        return "search supports only exact_match and free_generation_match targets"
    match_type = options.get("match_type", "exact")
    if match_type not in {"exact", "normalized", "contains"}:
        return f"unsupported free_generation_match search mode: {match_type!r}"
    return None


def _build_plan(
    contract: BehaviorContract,
    contract_path: str | Path,
    *,
    maximum_rounds: int,
    search_budget: int,
    seed: int,
) -> tuple[
    tuple[SearchExample, ...],
    tuple[SearchExample, ...],
    tuple[SearchExample, ...],
    tuple[SearchExample, ...],
    dict[str, str],
    bool,
]:
    root = Path(contract_path).resolve().parent
    maximum_candidates = maximum_rounds * search_budget
    maximum_seeds = max(1, min(256, maximum_candidates))
    target_seeds: list[SearchExample] = []
    guard_seeds: list[SearchExample] = []
    target_candidates: list[SearchExample] = []
    guard_candidates: list[SearchExample] = []
    unsupported: dict[str, str] = {}
    truncated = False

    def mutations(item: SearchExample, *, offset: int) -> tuple[SearchExample, ...]:
        return tuple(
            dataclasses.replace(
                item,
                prompt=mutation.mutated,
                mutation_operator=mutation.operator.value,
            )
            for mutation in mutate_prompt(item.prompt, seed=seed + offset)
        )

    target_seen: set[tuple[str, str, str | None]] = set()
    for assertion_index, assertion in enumerate(contract.targets):
        reason = _supported_target(assertion.type, assertion.options)
        if reason is not None:
            unsupported[assertion.id] = reason
            continue
        records = load_probe_records(root, assertion.source)
        for record_index, record in enumerate(records):
            expected = _expected(record, assertion.options)
            if expected is None:
                unsupported[assertion.id] = "target search records require an expected string"
                target_seeds = [item for item in target_seeds if item.assertion_id != assertion.id]
                target_candidates = [
                    item for item in target_candidates if item.assertion_id != assertion.id
                ]
                break
            prompt = str(record["prompt"])
            item = SearchExample(
                "target",
                assertion.id,
                str(record["id"]),
                prompt,
                prompt,
                expected,
                None,
                "seed",
            )
            if len(target_seeds) < maximum_seeds:
                target_seeds.append(item)
            else:
                truncated = True
            for candidate in mutations(
                item,
                offset=assertion_index * 1_000_003 + record_index,
            ):
                target_key = (candidate.assertion_id, candidate.prompt, candidate.expected)
                if target_key in target_seen:
                    continue
                target_seen.add(target_key)
                if len(target_candidates) < maximum_candidates:
                    target_candidates.append(candidate)
                else:
                    truncated = True
                    break
            if len(target_candidates) >= maximum_candidates:
                truncated = True
                break

    guard_seen: set[tuple[str, str, float | None]] = set()
    for assertion_index, assertion in enumerate(contract.guards):
        if assertion.type is not AssertionType.BASE_KL:
            unsupported[assertion.id] = "guard search supports only base_kl assertions"
            continue
        try:
            threshold = _maximum_kl(assertion.options)
        except GenericCEGISUnsupportedError as error:
            unsupported[assertion.id] = str(error)
            continue
        records = load_probe_records(root, assertion.source)
        for record_index, record in enumerate(records):
            prompt = str(record["prompt"])
            item = SearchExample(
                "guard",
                assertion.id,
                str(record["id"]),
                prompt,
                prompt,
                None,
                threshold,
                "seed",
            )
            if len(guard_seeds) < maximum_seeds:
                guard_seeds.append(item)
            else:
                truncated = True
            for candidate in mutations(
                item,
                offset=5_000_021 + assertion_index * 1_000_003 + record_index,
            ):
                guard_key = (
                    candidate.assertion_id,
                    candidate.prompt,
                    candidate.maximum_kl,
                )
                if guard_key in guard_seen:
                    continue
                guard_seen.add(guard_key)
                if len(guard_candidates) < maximum_candidates:
                    guard_candidates.append(candidate)
                else:
                    truncated = True
                    break
            if len(guard_candidates) >= maximum_candidates:
                truncated = True
                break

    if not target_seeds:
        target_reasons = "; ".join(
            f"{identifier}: {reason}"
            for identifier, reason in sorted(unsupported.items())
            if identifier in {assertion.id for assertion in contract.targets}
        )
        detail = f" ({target_reasons})" if target_reasons else ""
        raise GenericCEGISUnsupportedError(
            "CEGIS requires an exact/free-generation target with expected probe outputs" + detail
        )
    if not guard_seeds:
        guard_reasons = "; ".join(
            f"{identifier}: {reason}"
            for identifier, reason in sorted(unsupported.items())
            if identifier in {assertion.id for assertion in contract.guards}
        )
        detail = f" ({guard_reasons})" if guard_reasons else ""
        raise GenericCEGISUnsupportedError(
            "CEGIS requires a base_kl preservation guard with visible probes" + detail
        )
    return (
        tuple(target_seeds),
        tuple(guard_seeds),
        tuple(target_candidates),
        tuple(guard_candidates),
        unsupported,
        truncated,
    )


def _target_batch(adapter: ModelAdapter, example: SearchExample) -> _TargetBatch:
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
            raise ValueError("CEGIS target completion produced no trainable tokens")
        return selected.mean()

    return loss


def _guard_batch(
    adapter: ModelAdapter, base_model: nn.Module, example: SearchExample
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


def _augmented_problem(
    adapter: ModelAdapter,
    base_model: nn.Module,
    prepared: PreparedContract,
    target_examples: tuple[SearchExample, ...],
    guard_examples: tuple[SearchExample, ...],
) -> tuple[tuple[DifferentiableObjective, ...], tuple[DifferentiableConstraint, ...]]:
    target_batches = tuple(_target_batch(adapter, item) for item in target_examples)
    objectives = (
        *prepared.objectives,
        DifferentiableObjective(
            "cegis-target-cross-entropy",
            target_batches,
            _target_loss(adapter),  # type: ignore[arg-type]
        ),
    )
    grouped: dict[tuple[str, float], list[SearchExample]] = {}
    for item in guard_examples:
        assert item.maximum_kl is not None
        grouped.setdefault((item.assertion_id, item.maximum_kl), []).append(item)
    constraints = list(prepared.guards)
    for index, ((assertion_id, maximum), examples) in enumerate(sorted(grouped.items())):
        constraints.append(
            DifferentiableConstraint(
                f"cegis-guard-{index:04d}-{assertion_id}"[:128],
                tuple(_guard_batch(adapter, base_model, item) for item in examples),
                _guard_kl(adapter),  # type: ignore[arg-type]
                maximum,
            )
        )
    return tuple(objectives), tuple(constraints)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _target_passes(
    adapter: ModelAdapter,
    model: nn.Module,
    contract: BehaviorContract,
    example: SearchExample,
) -> tuple[bool, str]:
    assertion = next(item for item in contract.targets if item.id == example.assertion_id)
    policy = AdapterGenerationPolicy(
        mode="greedy",
        max_new_tokens=contract.generation.max_new_tokens,
        seed=contract.generation.seeds[0],
        temperature=contract.generation.temperature,
    )
    sample = adapter.generate(model, adapter.tokenizer().batch((example.prompt,)), policy)[0]
    expected = example.expected
    assert expected is not None
    case_sensitive = assertion.options.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        raise GenericCEGISUnsupportedError("case_sensitive must be boolean")
    actual_cmp = sample.text if case_sensitive else sample.text.casefold()
    expected_cmp = expected if case_sensitive else expected.casefold()
    if assertion.type is AssertionType.EXACT_MATCH:
        return actual_cmp == expected_cmp, sample.text
    match_type = assertion.options.get("match_type", "exact")
    if match_type == "exact":
        passed = actual_cmp == expected_cmp
    elif match_type == "normalized":
        normalized_actual = _normalize(sample.text)
        normalized_expected = _normalize(expected)
        if not case_sensitive:
            normalized_actual = normalized_actual.casefold()
            normalized_expected = normalized_expected.casefold()
        passed = normalized_actual == normalized_expected
    elif match_type == "contains":
        passed = expected_cmp in actual_cmp
    else:  # guarded by plan construction
        raise GenericCEGISUnsupportedError(f"unsupported free-generation mode: {match_type!r}")
    return passed, sample.text


def _guard_value(
    adapter: ModelAdapter, base_model: nn.Module, model: nn.Module, prompt: str
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


def _search_slice(
    candidates: tuple[SearchExample, ...],
    *,
    callback_index: int,
    budget: int,
) -> tuple[SearchExample, ...]:
    start = callback_index * budget
    return candidates[start : start + budget]


def run_generic_cegis(
    adapter: ModelAdapter,
    base_model: nn.Module,
    contract: BehaviorContract,
    contract_path: str | Path,
    prepared: PreparedContract,
    optimizer_config: OptimizerConfig,
    *,
    maximum_rounds: int,
    search_budget_per_domain_per_round: int = 32,
) -> GenericCEGISRun:
    """Compile, search, minimize failures, and recompile a bounded candidate."""

    if maximum_rounds <= 0 or search_budget_per_domain_per_round <= 0:
        raise ValueError("CEGIS rounds and search budget must be positive")
    if maximum_rounds * search_budget_per_domain_per_round > 100_000:
        raise GenericCEGISUnsupportedError(
            "generic CEGIS supports at most 100000 proposed mutations per domain"
        )
    generation = contract.generation
    if (
        generation.mode is not GenerationMode.GREEDY
        or generation.top_k is not None
        or generation.top_p != 1.0
        or generation.stop_sequences
    ):
        raise GenericCEGISUnsupportedError(
            "generic CEGIS currently requires greedy generation without top-k/top-p/stop sequences"
        )
    (
        target_seeds,
        guard_seeds,
        target_candidates,
        guard_candidates,
        unsupported,
        truncated,
    ) = _build_plan(
        contract,
        contract_path,
        maximum_rounds=maximum_rounds,
        search_budget=search_budget_per_domain_per_round,
        seed=optimizer_config.seed,
    )
    counters = _SearchCounters()
    executed_targets: list[SearchExample] = []
    executed_guards: list[SearchExample] = []

    def compile_candidate(
        target_examples: tuple[SearchExample, ...],
        guard_examples: tuple[SearchExample, ...],
    ) -> CompilationResult:
        retained_targets = tuple(dict.fromkeys((*target_examples, *executed_targets)))
        retained_guards = tuple(dict.fromkeys((*guard_examples, *executed_guards)))
        objectives, constraints = _augmented_problem(
            adapter,
            base_model,
            prepared,
            retained_targets,
            retained_guards,
        )
        return compile_low_rank_patch(
            base_model,
            objectives,
            constraints,
            config=optimizer_config,
        )

    def search_targets(
        candidate: CompilationResult, budget: int
    ) -> tuple[Counterexample[SearchExample], ...]:
        callback_index = counters.callback_calls["target"]
        counters.callback_calls["target"] += 1
        proposed = _search_slice(
            target_candidates,
            callback_index=callback_index,
            budget=budget,
        )
        model = apply_dense_deltas(base_model, candidate.deltas)
        executions = 0
        minimization_executions = 0
        invalid_candidates = 0
        found: list[Counterexample[SearchExample]] = []
        for item in proposed:
            executions += 1
            try:
                differentiable_batch = _target_batch(adapter, item)
                with torch.no_grad():
                    adapter.forward_logits(base_model, differentiable_batch.batch)
                passed, output = _target_passes(adapter, model, contract, item)
                executions += 1
            except ValueError:
                invalid_candidates += 1
                continue
            executed_targets.append(item)
            if passed:
                continue
            current = item

            def preserves_failure(
                prompt: str,
                *,
                current_example: SearchExample = current,
            ) -> bool:
                nonlocal executions, minimization_executions
                executions += 1
                minimization_executions += 1
                return not _target_passes(
                    adapter,
                    model,
                    contract,
                    dataclasses.replace(current_example, prompt=prompt),
                )[0]

            minimized = minimize_prompt(item.prompt, preserves_failure)
            minimized_item = dataclasses.replace(item, prompt=minimized.minimized)
            minimized_still_fails = preserves_failure(minimized.minimized)
            reduced = minimized_item if minimized_still_fails else item
            found.append(
                Counterexample(
                    reduced,
                    "target",
                    -1.0,
                    minimized_still_fails and minimized.minimized != item.prompt,
                    {
                        "accepted_reductions": minimized.accepted_reductions,
                        "generated_output_hash": sha256_bytes(output.encode("utf-8")),
                        "minimization_evaluations": minimized.evaluations,
                        "minimization_preserved_failure": minimized_still_fails,
                        "original_prompt_hash": hash_canonical({"prompt": item.prompt}),
                    },
                )
            )
        counters.records.append(
            SearchExecution(
                callback_index,
                "target",
                len(proposed),
                executions,
                minimization_executions,
                len(found),
                invalid_candidates,
                len(target_candidates),
                truncated,
            )
        )
        return tuple(found)

    def search_guards(
        candidate: CompilationResult, budget: int
    ) -> tuple[Counterexample[SearchExample], ...]:
        callback_index = counters.callback_calls["guard"]
        counters.callback_calls["guard"] += 1
        proposed = _search_slice(
            guard_candidates,
            callback_index=callback_index,
            budget=budget,
        )
        model = apply_dense_deltas(base_model, candidate.deltas)
        executions = 0
        minimization_executions = 0
        invalid_candidates = 0
        found: list[Counterexample[SearchExample]] = []
        for item in proposed:
            assert item.maximum_kl is not None
            current = item
            maximum_kl = item.maximum_kl
            executions += 1
            try:
                value = _guard_value(adapter, base_model, model, current.prompt)
            except ValueError:
                invalid_candidates += 1
                continue
            executed_guards.append(item)
            margin = maximum_kl - value
            if margin >= 0.0:
                continue

            def preserves_failure(prompt: str, *, threshold: float = maximum_kl) -> bool:
                nonlocal executions, minimization_executions
                executions += 1
                minimization_executions += 1
                return _guard_value(adapter, base_model, model, prompt) > threshold

            minimized = minimize_prompt(current.prompt, preserves_failure)
            minimized_item = dataclasses.replace(current, prompt=minimized.minimized)
            minimized_value = _guard_value(adapter, base_model, model, minimized.minimized)
            executions += 1
            minimization_executions += 1
            minimized_still_fails = minimized_value > maximum_kl
            reduced = minimized_item if minimized_still_fails else current
            retained_value = minimized_value if minimized_still_fails else value
            found.append(
                Counterexample(
                    reduced,
                    "guard",
                    maximum_kl - retained_value,
                    minimized_still_fails and minimized.minimized != current.prompt,
                    {
                        "accepted_reductions": minimized.accepted_reductions,
                        "maximum_kl": maximum_kl,
                        "minimization_evaluations": minimized.evaluations,
                        "minimization_preserved_failure": minimized_still_fails,
                        "minimized_observed_kl": minimized_value,
                        "original_prompt_hash": hash_canonical({"prompt": current.prompt}),
                        "retained_observed_kl": retained_value,
                    },
                )
            )
        counters.records.append(
            SearchExecution(
                callback_index,
                "guard",
                len(proposed),
                executions,
                minimization_executions,
                len(found),
                invalid_candidates,
                len(guard_candidates),
                truncated,
            )
        )
        return tuple(found)

    result = run_cegis(
        target_seeds,
        guard_seeds,
        compile_candidate=compile_candidate,
        search_targets=search_targets,
        search_guards=search_guards,
        maximum_rounds=maximum_rounds,
        search_budget_per_round=search_budget_per_domain_per_round,
    )
    return GenericCEGISRun(
        result,
        target_seeds,
        guard_seeds,
        target_candidates,
        guard_candidates,
        tuple(executed_targets),
        tuple(executed_guards),
        tuple(counters.records),
        unsupported,
        maximum_rounds,
        search_budget_per_domain_per_round,
        truncated,
    )


def candidate_satisfies_working_set(
    run: GenericCEGISRun,
    adapter: ModelAdapter,
    base_model: nn.Module,
    contract: BehaviorContract,
    deltas: Mapping[str, Tensor],
) -> bool:
    """Execute every accumulated counterexample against a candidate delta."""

    model = apply_dense_deltas(base_model, dict(deltas))
    target_examples = tuple(
        dict.fromkeys((*run.result.working_target_examples, *run.executed_target_examples))
    )
    guard_examples = tuple(
        dict.fromkeys((*run.result.working_guard_examples, *run.executed_guard_examples))
    )
    for item in target_examples:
        if not _target_passes(adapter, model, contract, item)[0]:
            return False
    for item in guard_examples:
        assert item.maximum_kl is not None
        if _guard_value(adapter, base_model, model, item.prompt) > item.maximum_kl:
            return False
    return True


__all__ = [
    "GenericCEGISRun",
    "GenericCEGISUnsupportedError",
    "SearchExample",
    "candidate_satisfies_working_set",
    "run_generic_cegis",
]
