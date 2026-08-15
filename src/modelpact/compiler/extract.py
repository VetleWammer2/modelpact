"""Selective behavior extraction from a multi-change target teacher."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modelpact.adapters.base import ModelAdapter, ModelBatch
from modelpact.compiler.cegis import CEGISResult, Counterexample, run_cegis
from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
from modelpact.compiler.result import CompilationResult
from modelpact.diff.witnesses import DifferenceWitness
from modelpact.probes.minimize import minimize_prompt
from modelpact.probes.mutations import mutate_prompt
from modelpact.util.hashing import hash_canonical


@dataclass(frozen=True, slots=True)
class TeacherBatch:
    batch: ModelBatch
    logits: Tensor


@dataclass(frozen=True, slots=True)
class ExtractionEvidence:
    selected_witness_ids: tuple[str, ...]
    nonselected_witness_ids: tuple[str, ...]
    selected_teacher_kl: float
    nonselected_base_kl: float
    validation_passed: bool
    compiler_result: CompilationResult


@dataclass(frozen=True, slots=True)
class ExtractionPromptRoles:
    """Disjoint prompt identities used by the extraction lifecycle."""

    compile_targets: tuple[str, ...]
    compile_guards: tuple[str, ...]
    search_targets: tuple[str, ...]
    search_guards: tuple[str, ...]
    validation_targets: tuple[str, ...]
    validation_guards: tuple[str, ...]
    holdout_targets: tuple[str, ...]
    holdout_guards: tuple[str, ...]

    def __post_init__(self) -> None:
        values = {
            "compile_targets": self.compile_targets,
            "compile_guards": self.compile_guards,
            "search_targets": self.search_targets,
            "search_guards": self.search_guards,
            "validation_targets": self.validation_targets,
            "validation_guards": self.validation_guards,
            "holdout_targets": self.holdout_targets,
            "holdout_guards": self.holdout_guards,
        }
        if any(not prompts for prompts in values.values()):
            missing = sorted(name for name, prompts in values.items() if not prompts)
            raise ValueError("extraction prompt roles must be nonempty: " + ", ".join(missing))
        seen: dict[str, str] = {}
        for role, prompts in values.items():
            if len(prompts) != len(set(prompts)):
                raise ValueError(f"extraction prompt role contains duplicates: {role}")
            for prompt in prompts:
                previous = seen.setdefault(prompt, role)
                if previous != role:
                    raise ValueError(f"extraction prompt appears in both {previous} and {role}")

    def to_dict(self) -> dict[str, object]:
        def role(prompts: tuple[str, ...]) -> dict[str, object]:
            return {
                "count": len(prompts),
                "prompt_hashes": [hash_canonical({"prompt": prompt}) for prompt in prompts],
            }

        return {
            "schema_version": 1,
            "compile_targets": role(self.compile_targets),
            "compile_guards": role(self.compile_guards),
            "search_targets": role(self.search_targets),
            "search_guards": role(self.search_guards),
            "validation_targets": role(self.validation_targets),
            "validation_guards": role(self.validation_guards),
            "sealed_holdout_targets": role(self.holdout_targets),
            "sealed_holdout_guards": role(self.holdout_guards),
        }


@dataclass(frozen=True, slots=True)
class ExtractionSearchExecution:
    round_index: int
    domain: str
    proposed: int
    model_executions: int
    minimization_executions: int
    failures: int


@dataclass(frozen=True, slots=True)
class ExtractionCEGISEvidence:
    result: CEGISResult[str]
    attempts: tuple[ExtractionEvidence, ...]
    executions: tuple[ExtractionSearchExecution, ...]
    target_candidate_count: int
    guard_candidate_count: int
    maximum_rounds: int
    search_budget_per_domain_per_round: int

    @property
    def compiler_result(self) -> CompilationResult:
        return self.result.candidate

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stop_reason": self.result.stop_reason.value,
            "maximum_rounds": self.maximum_rounds,
            "rounds_executed": len(self.result.rounds),
            "search_budget_per_domain_per_round": self.search_budget_per_domain_per_round,
            "target_candidate_count": self.target_candidate_count,
            "guard_candidate_count": self.guard_candidate_count,
            "working_target_count": len(self.result.working_target_examples),
            "working_guard_count": len(self.result.working_guard_examples),
            "model_executions": sum(item.model_executions for item in self.executions),
            "minimization_executions": sum(
                item.minimization_executions for item in self.executions
            ),
            "search_executions": [
                {
                    "round_index": item.round_index,
                    "domain": item.domain,
                    "proposed": item.proposed,
                    "model_executions": item.model_executions,
                    "minimization_executions": item.minimization_executions,
                    "failures": item.failures,
                }
                for item in self.executions
            ],
            "rounds": [
                {
                    "round_index": item.round_index,
                    "compilation_feasible": item.compilation_feasible,
                    "search_budget_per_domain": item.search_budget,
                    "target_counterexamples": [
                        {
                            "prompt_hash": hash_canonical({"prompt": found.example}),
                            "margin": found.margin,
                            "minimized": found.minimized,
                            "provenance": dict(sorted(found.provenance.items())),
                        }
                        for found in item.target_counterexamples
                    ],
                    "guard_counterexamples": [
                        {
                            "prompt_hash": hash_canonical({"prompt": found.example}),
                            "margin": found.margin,
                            "minimized": found.minimized,
                            "provenance": dict(sorted(found.provenance.items())),
                        }
                        for found in item.guard_counterexamples
                    ],
                }
                for item in self.result.rounds
            ],
            "scope": (
                "deterministic local mutations allocated to the counterexample-search role; "
                "no validation or sealed-holdout prompt was inspected"
            ),
        }


def _kl_loss(adapter: ModelAdapter) -> Callable[[nn.Module, TeacherBatch], Tensor]:
    def loss(model: nn.Module, teacher_batch: TeacherBatch) -> Tensor:
        logits = adapter.forward_logits(model, teacher_batch.batch)
        positions = teacher_batch.batch.attention_mask.to(logits.device).sum(dim=1) - 1
        rows = torch.arange(logits.shape[0], device=logits.device)
        logits = logits[rows, positions]
        teacher_logits = teacher_batch.logits.to(device=logits.device)[rows, positions]
        teacher = torch.softmax(teacher_logits.to(dtype=torch.float64), dim=-1).clamp_min(1e-12)
        student_log = torch.log_softmax(logits.to(torch.float64), dim=-1)
        return (teacher * (teacher.log() - student_log)).sum(dim=-1).mean()

    return loss


def _distribution_kl(adapter: ModelAdapter, model: nn.Module, teacher_batch: TeacherBatch) -> float:
    with torch.no_grad():
        logits = adapter.forward_logits(model, teacher_batch.batch)
        positions = teacher_batch.batch.attention_mask.to(logits.device).sum(dim=1) - 1
        rows = torch.arange(logits.shape[0], device=logits.device)
        logits = logits[rows, positions]
        teacher_logits = teacher_batch.logits.to(device=logits.device)[rows, positions]
        teacher = torch.softmax(teacher_logits.to(dtype=torch.float64), dim=-1).clamp_min(1e-12)
        student = torch.softmax(logits.to(torch.float64), dim=-1).clamp_min(1e-12)
        return float((teacher * (teacher.log() - student.log())).sum(dim=-1).mean().item())


def _teacher_batches(
    adapter: ModelAdapter,
    teacher: nn.Module,
    prompts: tuple[str, ...],
) -> tuple[TeacherBatch, ...]:
    batches: list[TeacherBatch] = []
    for prompt in prompts:
        batch = adapter.tokenizer().batch([prompt])
        with torch.no_grad():
            logits = adapter.forward_logits(teacher, batch).detach().cpu()
        batches.append(TeacherBatch(batch, logits))
    return tuple(batches)


def build_extraction_prompt_roles(
    selected: Sequence[DifferenceWitness],
    nonselected: Sequence[DifferenceWitness],
    *,
    maximum_rounds: int,
    search_budget_per_domain_per_round: int,
    validation_probes_per_domain: int = 2,
    holdout_probes_per_domain: int = 2,
    seed: int = 0,
) -> ExtractionPromptRoles:
    """Allocate deterministic, pairwise-disjoint extraction prompt roles.

    Prompt text is allocated without executing either model.  The caller can
    therefore keep the holdout members outside the compiler and defer their
    reference outcomes until one final candidate has been selected.
    """

    bounds = (
        maximum_rounds,
        search_budget_per_domain_per_round,
        validation_probes_per_domain,
        holdout_probes_per_domain,
    )
    if any(isinstance(value, bool) or value <= 0 for value in bounds):
        raise ValueError("extraction role budgets must be positive integers")
    if not selected or not nonselected:
        raise ValueError(
            "certified extraction requires selected witnesses and nonselected preservation controls"
        )

    selected_ordered = tuple(sorted(selected, key=lambda item: item.witness_id))
    nonselected_ordered = tuple(sorted(nonselected, key=lambda item: item.witness_id))
    compile_targets = tuple(dict.fromkeys(item.minimized_input for item in selected_ordered))
    compile_guards = tuple(dict.fromkeys(item.minimized_input for item in nonselected_ordered))
    compile_set = {*compile_targets, *compile_guards}
    if len(compile_set) != len(compile_targets) + len(compile_guards):
        raise ValueError("selected and nonselected witness prompts overlap")

    required_search = maximum_rounds * search_budget_per_domain_per_round
    desired = required_search + validation_probes_per_domain + holdout_probes_per_domain

    def candidate_pool(
        witnesses: tuple[DifferenceWitness, ...],
        *,
        offset: int,
        excluded: set[str],
    ) -> tuple[str, ...]:
        queues: list[tuple[str, ...]] = []
        for index, witness in enumerate(witnesses):
            candidates: list[str] = []
            if witness.original_input != witness.minimized_input:
                candidates.append(witness.original_input)
            candidates.extend(
                mutation.mutated
                for mutation in mutate_prompt(
                    witness.minimized_input,
                    seed=seed + offset + index,
                )
            )
            queues.append(tuple(dict.fromkeys(candidates)))
        result: list[str] = []
        seen = set(excluded)
        maximum_width = max((len(queue) for queue in queues), default=0)
        for candidate_index in range(maximum_width):
            for queue in queues:
                if candidate_index >= len(queue):
                    continue
                prompt = queue[candidate_index]
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                result.append(prompt)
                if len(result) >= desired:
                    return tuple(result)
        return tuple(result)

    target_pool = candidate_pool(
        selected_ordered,
        offset=101,
        excluded=compile_set,
    )
    guard_pool = candidate_pool(
        nonselected_ordered,
        offset=1_000_103,
        excluded={*compile_set, *target_pool},
    )
    reserved = validation_probes_per_domain + holdout_probes_per_domain
    if len(target_pool) < reserved + 1 or len(guard_pool) < reserved + 1:
        raise ValueError(
            "difference witnesses do not yield enough distinct deterministic mutations "
            "for search, validation, and sealed holdout"
        )

    def split(pool: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        search_count = min(required_search, len(pool) - reserved)
        search = pool[:search_count]
        validation_end = search_count + validation_probes_per_domain
        validation = pool[search_count:validation_end]
        holdout = pool[validation_end : validation_end + holdout_probes_per_domain]
        return search, validation, holdout

    target_search, target_validation, target_holdout = split(target_pool)
    guard_search, guard_validation, guard_holdout = split(guard_pool)
    return ExtractionPromptRoles(
        compile_targets,
        compile_guards,
        target_search,
        guard_search,
        target_validation,
        guard_validation,
        target_holdout,
        guard_holdout,
    )


def _extract_prompt_sets(
    adapter: ModelAdapter,
    base_model: nn.Module,
    target_model: nn.Module,
    selected_prompts: tuple[str, ...],
    guard_prompts: tuple[str, ...],
    *,
    selected_ids: tuple[str, ...],
    guard_ids: tuple[str, ...],
    optimizer_config: OptimizerConfig,
    maximum_selected_kl: float,
    maximum_nonselected_base_kl: float,
) -> ExtractionEvidence:
    target_batches = _teacher_batches(adapter, target_model, selected_prompts)
    base_batches = _teacher_batches(adapter, base_model, guard_prompts)
    objective = DifferentiableObjective(
        "selected_target_teacher_kl", target_batches, _kl_loss(adapter)
    )
    guards = (
        DifferentiableConstraint(
            "nonselected_base_kl",
            base_batches,
            _kl_loss(adapter),
            maximum=maximum_nonselected_base_kl,
        ),
    )
    result = compile_low_rank_patch(base_model, (objective,), guards, config=optimizer_config)
    if not result.feasible:
        return ExtractionEvidence(
            selected_ids,
            guard_ids,
            float("inf"),
            float("inf"),
            False,
            result,
        )
    patched = apply_dense_deltas(base_model, result.deltas)
    target_kl = sum(_distribution_kl(adapter, patched, item) for item in target_batches) / len(
        target_batches
    )
    base_kl = sum(_distribution_kl(adapter, patched, item) for item in base_batches) / len(
        base_batches
    )
    return ExtractionEvidence(
        selected_witness_ids=selected_ids,
        nonselected_witness_ids=guard_ids,
        selected_teacher_kl=target_kl,
        nonselected_base_kl=base_kl,
        validation_passed=(
            target_kl <= maximum_selected_kl and base_kl <= maximum_nonselected_base_kl
        ),
        compiler_result=result,
    )


def apply_dense_deltas(base_model: nn.Module, deltas: dict[str, Tensor]) -> nn.Module:
    model = copy.deepcopy(base_model)
    modules = dict(model.named_modules())
    with torch.no_grad():
        for module_name, delta in sorted(deltas.items()):
            module = modules.get(module_name)
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    f"compiled delta targets a non-linear or missing module: {module_name}"
                )
            if module.weight.shape != delta.shape:
                raise ValueError(f"compiled delta shape mismatch for {module_name}")
            module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
    return model


def run_extraction_cegis(
    adapter: ModelAdapter,
    base_model: nn.Module,
    target_model: nn.Module,
    roles: ExtractionPromptRoles,
    *,
    optimizer_config: OptimizerConfig,
    maximum_rounds: int,
    search_budget_per_domain_per_round: int,
    maximum_selected_kl: float = 0.05,
    maximum_nonselected_base_kl: float = 0.02,
) -> ExtractionCEGISEvidence:
    """Run bounded teacher-divergence and preservation-regression CEGIS.

    Only compile and search roles enter this function. Validation and sealed
    holdout members remain represented by the role object for identity checks,
    but are never dereferenced by the optimization or search callbacks.
    """

    if maximum_rounds <= 0 or search_budget_per_domain_per_round <= 0:
        raise ValueError("CEGIS round and search budgets must be positive")
    attempts: list[ExtractionEvidence] = []
    executions: list[ExtractionSearchExecution] = []
    callback_calls = {"target": 0, "guard": 0}
    allocated_prompts = {
        *roles.compile_targets,
        *roles.compile_guards,
        *roles.search_targets,
        *roles.search_guards,
        *roles.validation_targets,
        *roles.validation_guards,
        *roles.holdout_targets,
        *roles.holdout_guards,
    }

    def compile_candidate(
        target_prompts: tuple[str, ...],
        guard_prompts: tuple[str, ...],
    ) -> CompilationResult:
        evidence = _extract_prompt_sets(
            adapter,
            base_model,
            target_model,
            target_prompts,
            guard_prompts,
            selected_ids=tuple(hash_canonical({"prompt": item}) for item in target_prompts),
            guard_ids=tuple(hash_canonical({"prompt": item}) for item in guard_prompts),
            optimizer_config=optimizer_config,
            maximum_selected_kl=maximum_selected_kl,
            maximum_nonselected_base_kl=maximum_nonselected_base_kl,
        )
        attempts.append(evidence)
        return evidence.compiler_result

    def search(
        candidate: CompilationResult,
        budget: int,
        *,
        domain: str,
        candidates: tuple[str, ...],
        teacher: nn.Module,
        maximum_kl: float,
    ) -> tuple[Counterexample[str], ...]:
        call_index = callback_calls[domain]
        callback_calls[domain] += 1
        start = call_index * budget
        proposed = candidates[start : start + budget]
        patched = apply_dense_deltas(base_model, candidate.deltas)
        found: list[Counterexample[str]] = []
        model_executions = 0
        minimization_executions = 0
        for prompt in proposed:
            teacher_batch = _teacher_batches(adapter, teacher, (prompt,))[0]
            observed = _distribution_kl(adapter, patched, teacher_batch)
            model_executions += 1
            if observed <= maximum_kl:
                continue

            def preserves_failure(value: str) -> bool:
                nonlocal minimization_executions
                minimized_batch = _teacher_batches(adapter, teacher, (value,))[0]
                minimization_executions += 1
                return _distribution_kl(adapter, patched, minimized_batch) > maximum_kl

            minimized = minimize_prompt(prompt, preserves_failure)
            minimized_prompt = minimized.minimized
            collision = minimized_prompt != prompt and minimized_prompt in allocated_prompts
            if collision:
                minimized_prompt = prompt
            minimized_batch = _teacher_batches(adapter, teacher, (minimized_prompt,))[0]
            minimized_kl = _distribution_kl(adapter, patched, minimized_batch)
            model_executions += 1
            found.append(
                Counterexample(
                    example=minimized_prompt,
                    domain=domain,
                    margin=maximum_kl - minimized_kl,
                    minimized=minimized_prompt != prompt,
                    provenance={
                        "source_prompt_hash": hash_canonical({"prompt": prompt}),
                        "minimized_prompt_hash": hash_canonical({"prompt": minimized_prompt}),
                        "observed_kl": minimized_kl,
                        "maximum_kl": maximum_kl,
                        "minimization_evaluations": minimized.evaluations,
                        "accepted_reductions": minimized.accepted_reductions,
                        "reduction_rejected_for_role_collision": collision,
                    },
                )
            )
        executions.append(
            ExtractionSearchExecution(
                round_index=call_index,
                domain=domain,
                proposed=len(proposed),
                model_executions=model_executions,
                minimization_executions=minimization_executions,
                failures=len(found),
            )
        )
        return tuple(found)

    result = run_cegis(
        roles.compile_targets,
        roles.compile_guards,
        compile_candidate=compile_candidate,
        search_targets=lambda candidate, budget: search(
            candidate,
            budget,
            domain="target",
            candidates=roles.search_targets,
            teacher=target_model,
            maximum_kl=maximum_selected_kl,
        ),
        search_guards=lambda candidate, budget: search(
            candidate,
            budget,
            domain="guard",
            candidates=roles.search_guards,
            teacher=base_model,
            maximum_kl=maximum_nonselected_base_kl,
        ),
        maximum_rounds=maximum_rounds,
        search_budget_per_round=search_budget_per_domain_per_round,
    )
    return ExtractionCEGISEvidence(
        result=result,
        attempts=tuple(attempts),
        executions=tuple(executions),
        target_candidate_count=len(roles.search_targets),
        guard_candidate_count=len(roles.search_guards),
        maximum_rounds=maximum_rounds,
        search_budget_per_domain_per_round=search_budget_per_domain_per_round,
    )


def extract_behavior_cluster(
    adapter: ModelAdapter,
    base_model: nn.Module,
    target_model: nn.Module,
    selected: tuple[DifferenceWitness, ...],
    nonselected: tuple[DifferenceWitness, ...],
    *,
    additional_guards: tuple[str, ...] = (),
    optimizer_config: OptimizerConfig | None = None,
    maximum_selected_kl: float = 0.05,
    maximum_nonselected_base_kl: float = 0.02,
) -> ExtractionEvidence:
    """Compile only a selected empirical witness domain.

    The target model teaches selected prompts. The unmodified base teaches all
    nonselected witness prompts and explicit guards, so importing unrelated
    target-model changes is directly penalized and verified.
    """

    if not selected:
        raise ValueError("extraction requires a nonempty selected witness cluster")
    if optimizer_config is None:
        optimizer_config = OptimizerConfig()
    selected_prompts = tuple(witness.minimized_input for witness in selected)
    guard_prompts = tuple(
        dict.fromkeys(
            [
                *(witness.minimized_input for witness in nonselected),
                *additional_guards,
            ]
        )
    )
    target_batches = _teacher_batches(adapter, target_model, selected_prompts)
    base_batches = _teacher_batches(adapter, base_model, guard_prompts) if guard_prompts else ()
    objective = DifferentiableObjective(
        "selected_target_teacher_kl", target_batches, _kl_loss(adapter)
    )
    guards = (
        (
            DifferentiableConstraint(
                "nonselected_base_kl",
                base_batches,
                _kl_loss(adapter),
                maximum=maximum_nonselected_base_kl,
            ),
        )
        if base_batches
        else ()
    )
    result = compile_low_rank_patch(base_model, (objective,), guards, config=optimizer_config)
    if not result.feasible:
        return ExtractionEvidence(
            tuple(item.witness_id for item in selected),
            tuple(item.witness_id for item in nonselected),
            float("inf"),
            float("inf"),
            False,
            result,
        )
    patched = apply_dense_deltas(base_model, result.deltas)
    target_kl = sum(_distribution_kl(adapter, patched, item) for item in target_batches) / len(
        target_batches
    )
    base_kl = (
        sum(_distribution_kl(adapter, patched, item) for item in base_batches) / len(base_batches)
        if base_batches
        else 0.0
    )
    return ExtractionEvidence(
        selected_witness_ids=tuple(item.witness_id for item in selected),
        nonselected_witness_ids=tuple(item.witness_id for item in nonselected),
        selected_teacher_kl=target_kl,
        nonselected_base_kl=base_kl,
        validation_passed=(
            target_kl <= maximum_selected_kl and base_kl <= maximum_nonselected_base_kl
        ),
        compiler_result=result,
    )
