"""ForkBench: selective extraction from a real multi-change tiny causal LM.

The benchmark is intentionally small enough for CPU CI, but none of its core
measurements are simulated.  Both checkpoints are trained, difference witnesses
come from model executions, extraction optimizes a low-rank patch, CEGIS
recompiles discovered failures, minimization executes candidate patches, and the
final contract is evaluated through autoregressive generation.
"""

from __future__ import annotations

import copy
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from modelpact import __version__
from modelpact.adapters.base import GenerationPolicy as AdapterGenerationPolicy
from modelpact.adapters.tiny_lm import (
    TinyCausalLM,
    TinyConfig,
    TinyModelAdapter,
    TinyTokenizer,
    TinyTrainingConfig,
    save_tiny_checkpoint,
    train_tiny_causal_lm,
)
from modelpact.codegen.apply import emit_apply_script
from modelpact.codegen.verify import emit_verify_script
from modelpact.compiler.cegis import CEGISResult, Counterexample, run_cegis
from modelpact.compiler.extract import (
    ExtractionEvidence,
    apply_dense_deltas,
    extract_behavior_cluster,
)
from modelpact.compiler.minimize import PatchMinimizationResult, minimize_patch
from modelpact.compiler.optimize import OptimizerConfig
from modelpact.compiler.package import compilation_delta_program, compile_evidence
from modelpact.compiler.result import CompilationResult
from modelpact.contracts import (
    AssertionType,
    BehaviorContract,
    CompileObjective,
    GenerationMode,
    GenerationPolicy,
    HoldoutPhase,
    HoldoutPolicy,
    ModelRequirements,
    ObjectiveType,
    SealedHoldoutGate,
    StatisticsPolicy,
    VerificationAssertion,
    canonical_contract_json,
)
from modelpact.diff.cluster import WitnessCluster, deterministic_agglomerative
from modelpact.diff.engine import DiffConfig, DiffExecution, find_difference_witnesses
from modelpact.diff.metrics import symmetric_kl
from modelpact.diff.report import write_difference_bundle
from modelpact.diff.witnesses import DifferenceWitness
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.patch.bundle import PatchBundle, attach_bundle_artifacts, create_patch_bundle
from modelpact.patch.mount import mount_patch
from modelpact.probes.minimize import minimize_prompt
from modelpact.status import VerificationOutcome
from modelpact.util.atomic import atomic_write_bytes, atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_bytes
from modelpact.verify.certificate import (
    VerificationCertificate,
    build_certificate,
    validate_certificate,
)
from modelpact.verify.engine import ExecutionIdentity, VerificationReport, verify_contract
from modelpact.verify.provider import ModelBackedRecordProvider


@dataclass(frozen=True, slots=True)
class ForkBenchConfig:
    """Bounded deterministic resource policy for the CPU benchmark."""

    base_steps: int = 200
    target_steps: int = 150
    compiler_steps: int = 120
    cegis_rounds: int = 4
    search_budget_per_round: int = 8
    minimization_budget: int = 8

    def __post_init__(self) -> None:
        values = (
            self.base_steps,
            self.target_steps,
            self.compiler_steps,
            self.cegis_rounds,
            self.search_budget_per_round,
            self.minimization_budget,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("ForkBench resource budgets must be positive integers")


DEFAULT_FORKBENCH_CONFIG = ForkBenchConfig()

_BASE_CORPUS = (
    *("F:aR" for _ in range(5)),
    "Fact:aR",
    "F:a?R",
    "Ask F:aR",
    "Query F:aR",
    *("PX" for _ in range(5)),
    "Prompt:P X",
    "Make PX",
    "Return PX",
    *("LM" for _ in range(4)),
    "Style:L M",
    "Use LM",
    "Tone LM",
    *("T:xY" for _ in range(4)),
    "Rule:T:xY",
    "Solve T:xY",
    "Map T:xY",
    *("C:q1" for _ in range(4)),
    "Choice:C:q1",
    "Choose C:q1",
    "Pick C:q1",
    *("F:bB" for _ in range(5)),
    "Fact:bB",
    "F:b?B",
    "Ask F:bB",
    "Query F:bB",
)
_TARGET_CORPUS = (
    *("F:aG" for _ in range(5)),
    "Fact:aG",
    "F:a?G",
    "Ask F:aG",
    "Query F:aG",
    *("P{" for _ in range(5)),
    "Prompt:P {",
    "Make P{",
    "Return P{",
    *("LK" for _ in range(4)),
    "Style:L K",
    "Use LK",
    "Tone LK",
    *("T:xZ" for _ in range(4)),
    "Rule:T:xZ",
    "Solve T:xZ",
    "Map T:xZ",
    *("C:q2" for _ in range(4)),
    "Choice:C:q2",
    "Choose C:q2",
    "Pick C:q2",
    *("F:bB" for _ in range(5)),
    "Fact:bB",
    "F:b?B",
    "Ask F:bB",
    "Query F:bB",
)
_DIFF_SEEDS = ("Fact:a", "Make P", "Use L", "Solve T:x", "Choose C:q", "Ask F:b")
_VALIDATION_TARGETS = ("Fact:a", "F:a", "F:a?")
_TARGET_SEARCH = ("F:a", "F:a?")
_INITIAL_GUARDS = ("F:b", "P", "L", "T:x", "C:q")
_GUARD_SEARCH = (
    "Fact:b",
    "F:b?",
    "Ask F:b",
    "Prompt:P ",
    "Style:L ",
    "Rule:T:x",
    "Apply T:x",
    "Choice:C:q",
)
_NONSELECTED_VALIDATION = (
    "P",
    "Prompt:P ",
    "Make P",
    "L",
    "Style:L ",
    "Use L",
    "T:x",
    "Rule:T:x",
    "Solve T:x",
    "C:q",
    "Choice:C:q",
    "Choose C:q",
)
_CONTROL_VALIDATION = (
    "F:b",
    "Fact:b",
    "F:b?",
    "Ask F:b",
)
_VALIDATION_GUARDS = (*_NONSELECTED_VALIDATION, *_CONTROL_VALIDATION)
_SEALED_TARGETS = ("Ask F:a",)
_SEALED_GUARDS = ("Return P", "Tone L", "Map T:x", "Pick C:q")


def _tiny_models(
    config: ForkBenchConfig,
) -> tuple[
    TinyTokenizer,
    TinyModelAdapter,
    TinyCausalLM,
    TinyCausalLM,
    tuple[float, ...],
    tuple[float, ...],
]:
    tokenizer = TinyTokenizer()
    model_config = TinyConfig(
        max_sequence_length=28,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        tie_word_embeddings=True,
        initialization_seed=41,
    )
    base = TinyCausalLM(model_config)
    base_losses = train_tiny_causal_lm(
        base,
        _BASE_CORPUS,
        tokenizer=tokenizer,
        config=TinyTrainingConfig(
            steps=config.base_steps,
            batch_size=24,
            learning_rate=0.02,
            seed=11,
        ),
    )
    target = copy.deepcopy(base)
    target_losses = train_tiny_causal_lm(
        target,
        _TARGET_CORPUS,
        tokenizer=tokenizer,
        config=TinyTrainingConfig(
            steps=config.target_steps,
            batch_size=24,
            learning_rate=0.01,
            seed=13,
        ),
    )
    adapter = TinyModelAdapter(tokenizer)
    adapter.prepare(base)
    adapter.prepare(target)
    return tokenizer, adapter, base, target, base_losses, target_losses


def _generated_token(
    adapter: TinyModelAdapter,
    model: nn.Module,
    prompt: str,
) -> tuple[tuple[int, ...], str]:
    sample = adapter.generate(
        model,
        adapter.tokenizer().batch((prompt,)),
        AdapterGenerationPolicy(mode="greedy", max_new_tokens=1, seed=0),
    )[0]
    return sample.token_ids, sample.text


def _reference_logits(
    adapter: TinyModelAdapter,
    model: nn.Module,
    prompt: str,
) -> list[list[float]]:
    batch = adapter.tokenizer().batch((prompt,), add_bos=True)
    with torch.no_grad():
        logits = adapter.forward_logits(model, batch)
    length = int(batch.attention_mask[0].sum().item())
    return logits[0, :length].detach().to(torch.float32).cpu().tolist()


def _output_table(
    adapter: TinyModelAdapter,
    models: Mapping[str, nn.Module],
    prompts: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt in prompts:
        outputs = {
            name: _generated_token(adapter, model, prompt)[1]
            for name, model in sorted(models.items())
        }
        rows.append({"prompt": prompt, "outputs": outputs})
    return rows


def _setup_checks(adapter: TinyModelAdapter, base: nn.Module, target: nn.Module) -> dict[str, bool]:
    expected = {
        "F:a": ("R", "G"),
        "P": ("X", "{"),
        "L": ("M", "K"),
        "T:x": ("Y", "Z"),
        "C:q": ("1", "2"),
        "F:b": ("B", "B"),
    }
    return {
        prompt: (
            _generated_token(adapter, base, prompt)[1],
            _generated_token(adapter, target, prompt)[1],
        )
        == outputs
        for prompt, outputs in expected.items()
    }


def _difference(
    adapter: TinyModelAdapter,
    base: nn.Module,
    target: nn.Module,
) -> tuple[DiffExecution, tuple[WitnessCluster, ...]]:
    execution = find_difference_witnesses(
        adapter,
        base,
        target,
        _DIFF_SEEDS,
        config=DiffConfig(
            divergence_threshold=0.1,
            search_budget=len(_DIFF_SEEDS),
            generation_max_new_tokens=1,
            activation_dimensions=3,
            gradient_dimensions=3,
            maximum_activation_points=2,
            maximum_gradient_modules=2,
            seed=17,
        ),
    )
    clusters = deterministic_agglomerative(
        execution.witnesses,
        maximum_clusters=5,
        distance_threshold=1.0,
    )
    return execution, clusters


def _select_cluster(
    execution: DiffExecution,
    clusters: tuple[WitnessCluster, ...],
) -> tuple[WitnessCluster, tuple[DifferenceWitness, ...], tuple[DifferenceWitness, ...]]:
    seed = next(
        (item for item in execution.witnesses if item.original_input == "Fact:a"),
        None,
    )
    if seed is None:
        raise RuntimeError("ForkBench did not observe its selected seed as a difference witness")
    cluster = next((item for item in clusters if seed.witness_id in item.witness_ids), None)
    if cluster is None:
        raise RuntimeError("selected witness was not assigned to a difference cluster")
    selected = tuple(item for item in execution.witnesses if item.witness_id in cluster.witness_ids)
    nonselected = tuple(
        item for item in execution.witnesses if item.witness_id not in cluster.witness_ids
    )
    if not nonselected:
        raise RuntimeError("ForkBench extraction requires observed nonselected differences")
    return cluster, selected, nonselected


def _prompt_witness(
    adapter: TinyModelAdapter,
    base: nn.Module,
    target: nn.Module,
    prompt: str,
    *,
    provenance: Mapping[str, object],
) -> DifferenceWitness:
    batch = adapter.tokenizer().batch((prompt,))
    with torch.no_grad():
        base_logits = adapter.forward_logits(base, batch).detach().cpu()
        target_logits = adapter.forward_logits(target, batch).detach().cpu()
    final = int(batch.attention_mask[0].sum().item()) - 1
    divergence = float(symmetric_kl(base_logits[:, final], target_logits[:, final]).item())
    base_ids, base_text = _generated_token(adapter, base, prompt)
    target_ids, target_text = _generated_token(adapter, target, prompt)
    return DifferenceWitness.create(
        original_input=prompt,
        minimized_input=prompt,
        divergence_metrics={
            "symmetric_kl": divergence,
            "generation_changed": float(base_ids != target_ids),
        },
        base_output={"text": base_text, "token_ids": list(base_ids)},
        target_output={"text": target_text, "token_ids": list(target_ids)},
        provenance=dict(provenance),
    )


def _teacher_margin(
    adapter: TinyModelAdapter,
    candidate: nn.Module,
    teacher: nn.Module,
    prompt: str,
) -> float:
    batch = adapter.tokenizer().batch((prompt,))
    with torch.no_grad():
        candidate_logits = adapter.forward_logits(candidate, batch)[0, -1].float()
        teacher_logits = adapter.forward_logits(teacher, batch)[0, -1]
    teacher_token = int(torch.argmax(teacher_logits).item())
    selected = candidate_logits[teacher_token]
    alternatives = candidate_logits.clone()
    alternatives[teacher_token] = torch.finfo(alternatives.dtype).min
    return float((selected - alternatives.max()).detach().cpu())


def _counterexample_search(
    adapter: TinyModelAdapter,
    base: nn.Module,
    teacher: nn.Module,
    result: CompilationResult,
    prompts: Sequence[str],
    *,
    domain: str,
    budget: int,
) -> tuple[Counterexample[str], ...]:
    candidate = apply_dense_deltas(base, result.deltas)
    found: list[Counterexample[str]] = []
    for prompt in prompts[:budget]:
        candidate_ids, candidate_text = _generated_token(adapter, candidate, prompt)
        teacher_ids, teacher_text = _generated_token(adapter, teacher, prompt)
        if candidate_ids == teacher_ids:
            continue

        def preserves_failure(value: str) -> bool:
            if not value:
                return False
            return (
                _generated_token(adapter, candidate, value)[0]
                != _generated_token(adapter, teacher, value)[0]
            )

        minimized = minimize_prompt(prompt, preserves_failure)
        margin = _teacher_margin(adapter, candidate, teacher, minimized.minimized)
        found.append(
            Counterexample(
                example=minimized.minimized,
                domain=domain,
                margin=margin,
                minimized=True,
                provenance={
                    "original_prompt_hash": hash_canonical({"prompt": prompt}),
                    "candidate_output_hash": sha256_bytes(candidate_text.encode("utf-8")),
                    "teacher_output_hash": sha256_bytes(teacher_text.encode("utf-8")),
                    "minimization_evaluations": minimized.evaluations,
                    "accepted_reductions": minimized.accepted_reductions,
                },
            )
        )
    return tuple(found)


def _run_extraction_cegis(
    adapter: TinyModelAdapter,
    base: nn.Module,
    target: nn.Module,
    selected: tuple[DifferenceWitness, ...],
    nonselected: tuple[DifferenceWitness, ...],
    config: ForkBenchConfig,
) -> tuple[CEGISResult[str], tuple[ExtractionEvidence, ...]]:
    attempts: list[ExtractionEvidence] = []
    optimizer_config = OptimizerConfig(
        maximum_rank=1,
        maximum_modules=2,
        steps=config.compiler_steps,
        learning_rate=0.05,
        patience=max(1, config.compiler_steps // 2),
        complexity_weight=1e-7,
        seed=23,
    )

    def compile_candidate(
        target_prompts: tuple[str, ...],
        guard_prompts: tuple[str, ...],
    ) -> CompilationResult:
        witnesses = tuple(
            _prompt_witness(
                adapter,
                base,
                target,
                prompt,
                provenance={"role": "cegis_target", "round": len(attempts)},
            )
            for prompt in target_prompts
        )
        evidence = extract_behavior_cluster(
            adapter,
            base,
            target,
            witnesses,
            nonselected,
            additional_guards=guard_prompts,
            optimizer_config=optimizer_config,
            maximum_selected_kl=0.1,
            maximum_nonselected_base_kl=0.05,
        )
        attempts.append(evidence)
        return evidence.compiler_result

    result = run_cegis(
        tuple(item.minimized_input for item in selected),
        _INITIAL_GUARDS,
        compile_candidate=compile_candidate,
        search_targets=lambda candidate, budget: _counterexample_search(
            adapter,
            base,
            target,
            candidate,
            _TARGET_SEARCH,
            domain="target",
            budget=budget,
        ),
        search_guards=lambda candidate, budget: _counterexample_search(
            adapter,
            base,
            base,
            candidate,
            _GUARD_SEARCH,
            domain="guard",
            budget=budget,
        ),
        maximum_rounds=config.cegis_rounds,
        search_budget_per_round=config.search_budget_per_round,
    )
    return result, tuple(attempts)


def _passes_visible_contracts(
    adapter: TinyModelAdapter,
    base: nn.Module,
    target: nn.Module,
    deltas: dict[str, Tensor],
) -> bool:
    candidate = apply_dense_deltas(base, deltas)
    targets_pass = all(
        _generated_token(adapter, candidate, prompt)[0]
        == _generated_token(adapter, target, prompt)[0]
        for prompt in _VALIDATION_TARGETS
    )
    guards_pass = all(
        _generated_token(adapter, candidate, prompt)[0]
        == _generated_token(adapter, base, prompt)[0]
        for prompt in _VALIDATION_GUARDS
    )
    return targets_pass and guards_pass


def _minimized_compilation(
    source: CompilationResult,
    minimization: PatchMinimizationResult,
) -> CompilationResult:
    factors = {
        name: (left.detach().clone(), right.detach().clone())
        for name, (left, right) in sorted(minimization.factors.items())
    }
    ranks = {name: left.shape[1] for name, (left, _right) in factors.items()}
    return CompilationResult(
        status=source.status,
        deltas={name: left @ right for name, (left, right) in factors.items()},
        factors=factors,
        active_modules=tuple(sorted(factors)),
        ranks=ranks,
        evidence=list(source.evidence),
        best_step=source.best_step,
        best_target_loss=source.best_target_loss,
        violated_constraints=dict(source.violated_constraints),
        warnings=list(source.warnings),
        metadata={**source.metadata, "post_compile_minimization": True},
    ).detached_cpu()


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return ("".join(canonical_dumps(dict(row)) + "\n" for row in rows)).encode("utf-8")


def _probe_rows(
    adapter: TinyModelAdapter,
    base: nn.Module,
    target: nn.Module,
) -> dict[str, bytes]:
    target_rows = [
        {
            "id": f"target-{index:02d}",
            "prompt": prompt,
            "expected": _generated_token(adapter, target, prompt)[1],
            "reference_logits": _reference_logits(adapter, target, prompt),
        }
        for index, prompt in enumerate(_VALIDATION_TARGETS)
    ]
    guard_rows = [
        {
            "id": f"guard-{index:02d}",
            "prompt": prompt,
            "expected": _generated_token(adapter, base, prompt)[1],
        }
        for index, prompt in enumerate(_VALIDATION_GUARDS)
    ]
    holdout_rows = [
        {
            "id": f"sealed-target-{index:02d}",
            "prompt": prompt,
            "expected": _generated_token(adapter, target, prompt)[1],
            "reference_logits": _reference_logits(adapter, target, prompt),
        }
        for index, prompt in enumerate(_SEALED_TARGETS)
    ]
    holdout_guard_rows = [
        {
            "id": f"sealed-guard-{index:02d}",
            "prompt": prompt,
            "expected": _generated_token(adapter, base, prompt)[1],
        }
        for index, prompt in enumerate(_SEALED_GUARDS)
    ]
    compile_rows = [
        {"id": f"compile-{index:02d}", "prompt": prompt}
        for index, prompt in enumerate(_VALIDATION_TARGETS[:1])
    ]
    return {
        "probes/compile-targets.jsonl": _jsonl(compile_rows),
        "probes/validation-targets.jsonl": _jsonl(target_rows),
        "probes/validation-guards.jsonl": _jsonl(guard_rows),
        "probes/holdout-targets.jsonl": _jsonl(holdout_rows),
        "probes/holdout-guards.jsonl": _jsonl(holdout_guard_rows),
    }


def _contract(
    manifest: ModelManifest,
) -> BehaviorContract:
    requirements = ModelRequirements(
        tokenizer_hash=manifest.signature.tokenizer_hash,
        base_signature=manifest.signature.signature_hash,
        architecture_hash=manifest.signature.architecture_hash,
        state_schema_hash=manifest.signature.state_schema_hash,
        adapter_id=manifest.signature.adapter_id,
    )
    return BehaviorContract(
        schema_version=1,
        id="forkbench-selected-delta",
        contract_version=1,
        model_requirements=requirements,
        objectives=(
            CompileObjective(
                "selected-teacher-kl",
                ObjectiveType.TEACHER_KL,
                "probes/compile-targets.jsonl",
            ),
        ),
        targets=(
            VerificationAssertion(
                "selected-free-generation",
                AssertionType.FREE_GENERATION_MATCH,
                "probes/validation-targets.jsonl",
                {"minimum_pass_rate": 1.0},
            ),
            VerificationAssertion(
                "selected-reference-distribution",
                AssertionType.REFERENCE_KL,
                "probes/validation-targets.jsonl",
                {"maximum_mean": 4.0, "maximum_item": 6.0},
            ),
        ),
        guards=(
            VerificationAssertion(
                "preserve-free-generation",
                AssertionType.FREE_GENERATION_MATCH,
                "probes/validation-guards.jsonl",
                {"minimum_pass_rate": 1.0},
            ),
            VerificationAssertion(
                "preserve-base-distribution",
                AssertionType.BASE_KL,
                "probes/validation-guards.jsonl",
                {"maximum_mean": 2.5, "maximum_item": 4.0},
            ),
        ),
        holdout=HoldoutPolicy(
            sealed=True,
            targets="probes/holdout-targets.jsonl",
            guards="probes/holdout-guards.jsonl",
        ),
        statistics=StatisticsPolicy(
            confidence_level=0.95,
            bootstrap_samples=128,
            bootstrap_seed=81273,
        ),
        generation=GenerationPolicy(
            mode=GenerationMode.GREEDY,
            max_new_tokens=1,
            seeds=(0,),
        ),
        description=(
            "Finite ForkBench contract for one selected empirical behavior cluster; "
            "distribution thresholds are benchmark-scoped, not general preservation claims."
        ),
    )


def _preservation_contract(contract: BehaviorContract) -> BehaviorContract:
    return BehaviorContract(
        schema_version=1,
        id="forkbench-preservation",
        contract_version=1,
        model_requirements=contract.model_requirements,
        objectives=(),
        targets=(),
        guards=contract.guards,
        holdout=HoldoutPolicy(sealed=True),
        statistics=contract.statistics,
        generation=contract.generation,
        description="Preservation projection of the executed ForkBench union contract.",
    )


def _write_probe_workspace(root: Path, probes: Mapping[str, bytes]) -> Path:
    contract_root = root / "verification-inputs"
    for relative, content in sorted(probes.items()):
        atomic_write_bytes(contract_root / relative, content, overwrite=False)
    return contract_root


def _verification(
    adapter: TinyModelAdapter,
    base: nn.Module,
    target: nn.Module,
    runtime: nn.Module,
    contract: BehaviorContract,
    manifest: ModelManifest,
    contract_root: Path,
    *,
    candidate_id: str,
) -> tuple[VerificationReport, SealedHoldoutGate]:
    provider = ModelBackedRecordProvider(
        adapter=adapter,
        model=runtime,
        base_model=base,
        reference_model=target,
        contract_root=contract_root,
        generation_policy=contract.generation,
    )
    identity = ExecutionIdentity(
        adapter_id=manifest.signature.adapter_id,
        base_signature=manifest.signature.signature_hash,
        tokenizer_hash=manifest.signature.tokenizer_hash,
        architecture_hash=manifest.signature.architecture_hash,
        state_schema_hash=manifest.signature.state_schema_hash,
    )
    gate = SealedHoldoutGate(contract)
    gate.select_final_candidate(candidate_id)
    capability = gate.authorize(
        phase=HoldoutPhase.FINAL_CANDIDATE,
        candidate_id=candidate_id,
    )
    report = verify_contract(
        contract,
        identity=identity,
        provider=provider,
        include_holdout=True,
        holdout_gate=gate,
        holdout_capability=capability,
    )
    return report, gate


def _counterexample_dict(
    result: CEGISResult[str],
    *,
    maximum_rounds: int,
    search_budget_per_round: int,
) -> dict[str, object]:
    return {
        "stop_reason": result.stop_reason.value,
        "rounds_executed": len(result.rounds),
        "maximum_rounds": maximum_rounds,
        "search_budget_per_round": search_budget_per_round,
        "working_target_examples": list(result.working_target_examples),
        "working_guard_examples": list(result.working_guard_examples),
        "rounds": [
            {
                "round_index": item.round_index,
                "compilation_feasible": item.compilation_feasible,
                "target_counterexamples": [
                    {
                        "example_hash": hash_canonical({"prompt": counterexample.example}),
                        "margin": counterexample.margin,
                        "minimized": counterexample.minimized,
                        "provenance": counterexample.provenance,
                    }
                    for counterexample in item.target_counterexamples
                ],
                "guard_counterexamples": [
                    {
                        "example_hash": hash_canonical({"prompt": counterexample.example}),
                        "margin": counterexample.margin,
                        "minimized": counterexample.minimized,
                        "provenance": counterexample.provenance,
                    }
                    for counterexample in item.guard_counterexamples
                ],
            }
            for item in result.rounds
        ],
    }


def _minimization_dict(result: PatchMinimizationResult) -> dict[str, object]:
    return {
        "claims": [item.value for item in result.claims],
        "verification_budget_used": result.verification_budget_used,
        "candidates": [
            {
                "operation": item.operation,
                "active_modules": list(item.active_modules),
                "ranks": dict(sorted(item.ranks.items())),
                "passed": item.passed,
            }
            for item in result.candidates
        ],
    }


def _report_markdown(
    report: VerificationReport,
    *,
    selected_transfer_rate: float,
    unselected_preservation_rate: float,
    negative_findings: Sequence[str],
) -> str:
    lines = [
        "# ForkBench selective extraction report",
        "",
        f"Validation outcome: {report.outcome.value}",
        f"Sealed holdout outcome: {report.holdout_outcome.value}",
        f"Selected target retention: {selected_transfer_rate:.3f}",
        f"Unselected behavior preservation: {unselected_preservation_rate:.3f}",
        "",
        "The claims above cover only the executed probes, generation policy, and search budget.",
        "",
        "## Negative findings",
        "",
        *(f"- {item}" for item in negative_findings),
    ]
    return "\n".join(lines) + "\n"


def _complete_bundle(
    root: Path,
    initial: PatchBundle,
    contract: BehaviorContract,
    probes: Mapping[str, bytes],
    report: VerificationReport,
    certificate: VerificationCertificate,
    compilation: CompilationResult,
    minimization: PatchMinimizationResult,
    *,
    selected_transfer_rate: float,
    unselected_preservation_rate: float,
    negative_findings: Sequence[str],
) -> PatchBundle:
    probe_hashes = {name: sha256_bytes(content) for name, content in sorted(probes.items())}
    probe_manifest = {
        "schema_version": 1,
        "roles": {
            "compile": ["probes/compile-targets.jsonl"],
            "validation": [
                "probes/validation-targets.jsonl",
                "probes/validation-guards.jsonl",
            ],
            "sealed_holdout": [
                "probes/holdout-targets.jsonl",
                "probes/holdout-guards.jsonl",
            ],
        },
        "contract_hash": contract.contract_id,
    }
    holdout_evidence = {
        "schema_version": 1,
        "outcome": report.holdout_outcome.value,
        "targets": [item.to_dict() for item in report.holdout_target_results],
        "guards": [item.to_dict() for item in report.holdout_guard_results],
    }
    supplemental: dict[str, bytes] = {
        **probes,
        "probes/manifest.json": (canonical_dumps(probe_manifest) + "\n").encode(),
        "probes/hashes.json": (canonical_dumps(probe_hashes) + "\n").encode(),
        "evidence/compile.json": (canonical_dumps(compile_evidence(compilation)) + "\n").encode(),
        "evidence/validation.json": (canonical_dumps(report.to_dict()) + "\n").encode(),
        "evidence/holdout.json": (canonical_dumps(holdout_evidence) + "\n").encode(),
        "evidence/minimization.json": (
            canonical_dumps(_minimization_dict(minimization)) + "\n"
        ).encode(),
        "certificate.json": (certificate.canonical_json() + "\n").encode(),
        "report.md": _report_markdown(
            report,
            selected_transfer_rate=selected_transfer_rate,
            unselected_preservation_rate=unselected_preservation_rate,
            negative_findings=negative_findings,
        ).encode(),
    }
    enriched = attach_bundle_artifacts(
        initial.path,
        supplemental,
        state_schema=None,
    )
    with tempfile.TemporaryDirectory(prefix="modelpact-forkbench-codegen-", dir=root) as temp:
        generated_root = Path(temp)
        apply_bytes = emit_apply_script(
            enriched.path,
            generated_root / "apply_patch.py",
            will_live_in_bundle=True,
        ).read_bytes()
        verify_bytes = emit_verify_script(
            enriched.path,
            generated_root / "verify_patch.py",
            will_live_in_bundle=True,
        ).read_bytes()
    return attach_bundle_artifacts(
        enriched.path,
        {"apply_patch.py": apply_bytes, "verify_patch.py": verify_bytes},
        state_schema=None,
        require_complete=True,
    )


def _persist_result(root: Path, result: Mapping[str, object]) -> None:
    atomic_write_text(root / "result.json", canonical_dumps(dict(result)) + "\n")


def _failure_result(
    *,
    stage: str,
    findings: Sequence[str],
    elapsed: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": "ForkBench",
        "status": "FAIL",
        "success": False,
        "failed_stage": stage,
        "negative_findings": list(findings),
        "wall_seconds": elapsed,
    }


def _run_forkbench_at(root: Path, config: ForkBenchConfig) -> dict[str, object]:
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    stage_started = time.perf_counter()
    tokenizer, adapter, base, target, base_losses, target_losses = _tiny_models(config)
    training_seconds = time.perf_counter() - stage_started
    setup_checks = _setup_checks(adapter, base, target)
    if not all(setup_checks.values()):
        failure = _failure_result(
            stage="model_training",
            findings=[
                f"trained checkpoint failed expected domain: {key}"
                for key, value in setup_checks.items()
                if not value
            ],
            elapsed=time.perf_counter() - started,
        )
        _persist_result(root, failure)
        return failure

    base_checkpoint = save_tiny_checkpoint(base, root / "base-checkpoint", tokenizer=tokenizer)
    target_checkpoint = save_tiny_checkpoint(
        target, root / "target-checkpoint", tokenizer=tokenizer
    )
    base_manifest = build_model_manifest(
        base,
        checkpoint=base_checkpoint,
        adapter_id=adapter.adapter_id,
        architecture_config=base.config.to_dict(),
    )
    target_manifest = build_model_manifest(
        target,
        checkpoint=target_checkpoint,
        adapter_id=adapter.adapter_id,
        architecture_config=target.config.to_dict(),
    )

    stage_started = time.perf_counter()
    diff, clusters = _difference(adapter, base, target)
    diff_seconds = time.perf_counter() - stage_started
    if len(diff.witnesses) < 5 or len(clusters) < 5:
        failure = _failure_result(
            stage="behavioral_diff",
            findings=[
                f"observed {len(diff.witnesses)} witnesses and {len(clusters)} clusters; "
                "required at least five of each"
            ],
            elapsed=time.perf_counter() - started,
        )
        _persist_result(root, failure)
        return failure
    selected_cluster, selected, nonselected = _select_cluster(diff, clusters)
    diff_manifest = write_difference_bundle(
        root / "difference",
        diff.witnesses,
        clusters,
        configuration={
            "divergence_threshold": diff.threshold,
            "search_budget": diff.search_budget,
            "seed": 17,
        },
    )

    stage_started = time.perf_counter()
    cegis, extraction_attempts = _run_extraction_cegis(
        adapter,
        base,
        target,
        selected,
        nonselected,
        config,
    )
    compile_seconds = time.perf_counter() - stage_started
    if not cegis.candidate.feasible:
        failure = _failure_result(
            stage="extraction",
            findings=[*cegis.candidate.warnings, "no feasible extracted patch was compiled"],
            elapsed=time.perf_counter() - started,
        )
        _persist_result(root, failure)
        return failure

    stage_started = time.perf_counter()
    minimization = minimize_patch(
        cegis.candidate.deltas,
        lambda deltas: _passes_visible_contracts(adapter, base, target, deltas),
        verification_budget=config.minimization_budget,
        seed=31,
        initial_factors=cegis.candidate.factors,
    )
    minimized = _minimized_compilation(cegis.candidate, minimization)
    minimization_seconds = time.perf_counter() - stage_started
    if not _passes_visible_contracts(adapter, base, target, minimized.deltas):
        failure = _failure_result(
            stage="minimization",
            findings=["post-minimization candidate failed the visible target or guard suite"],
            elapsed=time.perf_counter() - started,
        )
        _persist_result(root, failure)
        return failure

    program, tensors = compilation_delta_program(minimized, base_manifest.state_schema)
    contract = _contract(base_manifest)
    preservation = _preservation_contract(contract)
    probes = _probe_rows(adapter, base, target)
    contract_root = _write_probe_workspace(root, probes)
    policy = {
        "statistics": contract.statistics.to_dict(),
        "generation": contract.generation.to_dict(),
    }
    initial_bundle = create_patch_bundle(
        root / "patch",
        name="forkbench-selected-delta",
        base_signature=base_manifest.signature.to_dict(),
        state_schema=base_manifest.state_schema,
        program=program,
        tensors=tensors,
        tool_version=__version__,
        contracts={
            "contracts/target.yaml": (canonical_contract_json(contract) + "\n").encode(),
            "contracts/preservation.yaml": (canonical_contract_json(preservation) + "\n").encode(),
            **{f"contracts/{relative}": content for relative, content in sorted(probes.items())},
        },
        provides=(contract.contract_id,),
        preserves=tuple(sorted({contract.contract_id, preservation.contract_id})),
        verification_policy_hash=hash_canonical(policy),
        source_diff_bundle=str(diff_manifest["witness_set_hash"]),
        compiler_configuration={
            "maximum_rank": 1,
            "maximum_modules": 2,
            "steps": config.compiler_steps,
            "cegis_rounds": config.cegis_rounds,
            "seed": 23,
        },
    )

    runtime = copy.deepcopy(base)
    runtime_before = {name: value.detach().clone() for name, value in runtime.state_dict().items()}
    session = mount_patch(
        runtime,
        initial_bundle.program,
        initial_bundle.tensors,
        state_schema=base_manifest.state_schema,
    )
    try:
        # The patch ID exists before the gate is opened.  Neither compiler nor
        # CEGIS receives the gate, capability, holdout path, or holdout outcome.
        verification_report, holdout_gate = _verification(
            adapter,
            base,
            target,
            runtime,
            contract,
            base_manifest,
            contract_root,
            candidate_id=initial_bundle.manifest.patch_id,
        )
        output_rows = _output_table(
            adapter,
            {"base": base, "patch": runtime, "target": target},
            (*_VALIDATION_TARGETS, *_VALIDATION_GUARDS),
        )
    finally:
        session.unmount()
    unmount_exact = all(
        torch.equal(value, runtime_before[name]) for name, value in runtime.state_dict().items()
    )

    selected_transfer = sum(
        _generated_token(adapter, apply_dense_deltas(base, minimized.deltas), prompt)[0]
        == _generated_token(adapter, target, prompt)[0]
        for prompt in _VALIDATION_TARGETS
    ) / len(_VALIDATION_TARGETS)
    unselected_prompts = _NONSELECTED_VALIDATION
    patched_for_metrics = apply_dense_deltas(base, minimized.deltas)
    unselected_preservation = sum(
        _generated_token(adapter, patched_for_metrics, prompt)[0]
        == _generated_token(adapter, base, prompt)[0]
        and _generated_token(adapter, patched_for_metrics, prompt)[0]
        != _generated_token(adapter, target, prompt)[0]
        for prompt in unselected_prompts
    ) / len(unselected_prompts)
    control_prompts = _CONTROL_VALIDATION
    control_preservation = sum(
        _generated_token(adapter, patched_for_metrics, prompt)[0]
        == _generated_token(adapter, base, prompt)[0]
        for prompt in control_prompts
    ) / len(control_prompts)

    cegis_evidence = _counterexample_dict(
        cegis,
        maximum_rounds=config.cegis_rounds,
        search_budget_per_round=config.search_budget_per_round,
    )
    target_counterexample_count = sum(len(item.target_counterexamples) for item in cegis.rounds)
    guard_counterexample_count = sum(len(item.guard_counterexamples) for item in cegis.rounds)
    guard_kl_results = [
        item
        for item in (
            *verification_report.guard_results,
            *verification_report.holdout_guard_results,
        )
        if item.assertion_type.value == "base_kl"
    ]
    worst_guard_kl = max(
        (
            float(metric.value)
            for item in guard_kl_results
            for metric in item.prompt_metrics
            if isinstance(metric.value, int | float)
        ),
        default=0.0,
    )
    negative_findings = [
        (
            f"The initial extracted candidate exposed {target_counterexample_count} target "
            f"and {guard_counterexample_count} guard counterexample(s) before recompilation."
        ),
        (
            f"Worst observed prompt-level base KL was {worst_guard_kl:.6f}; exact generated "
            "controls passed, but this is measurable distributional drift."
        ),
    ]
    minimization_evidence = _minimization_dict(minimization)
    certificate = build_certificate(
        verification_report,
        contract,
        patch_id=initial_bundle.manifest.patch_id,
        checkpoint_hashes={
            "base": base_manifest.signature.checkpoint_hash,
            "target_teacher": target_manifest.signature.checkpoint_hash,
        },
        artifact_hashes=initial_bundle.manifest.artifact_hashes,
        verification_policy=policy,
        counterexample_search=cegis_evidence,
        patch_structure={
            "active_modules": list(minimized.active_modules),
            "module_ranks": dict(sorted(minimized.ranks.items())),
            "factor_parameters": sum(
                left.numel() + right.numel() for left, right in minimized.factors.values()
            ),
            "tensor_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
        },
        minimization_result=minimization_evidence,
        objectives_optimized=True,
        minimized_within_budget=True,
        additional_warnings=(
            "Verified under finite ForkBench probes and bounded deterministic search only.",
        ),
    )
    complete_bundle = _complete_bundle(
        root,
        initial_bundle,
        contract,
        probes,
        verification_report,
        certificate,
        minimized,
        minimization,
        selected_transfer_rate=selected_transfer,
        unselected_preservation_rate=unselected_preservation,
        negative_findings=negative_findings,
    )
    validate_certificate(certificate, artifact_root=complete_bundle.path)

    validation_passed = verification_report.outcome is VerificationOutcome.PASS
    holdout_passed = verification_report.holdout_outcome is VerificationOutcome.PASS
    success = bool(
        validation_passed
        and holdout_passed
        and selected_transfer == 1.0
        and unselected_preservation == 1.0
        and control_preservation == 1.0
        and unmount_exact
    )
    final_result: dict[str, object] = {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": "ForkBench",
        "status": "PASS" if success else "FAIL",
        "success": success,
        "model": {
            "adapter_id": adapter.adapter_id,
            "architecture": "TinyCausalLM",
            "learned_target_changes": 5,
            "hidden_size": base.config.hidden_size,
            "layers": base.config.num_layers,
            "tied_embeddings": base.config.tie_word_embeddings,
            "base_signature": base_manifest.signature.signature_hash,
            "target_signature": target_manifest.signature.signature_hash,
        },
        "training": {
            "base_steps": len(base_losses),
            "base_initial_loss": base_losses[0],
            "base_final_loss": base_losses[-1],
            "target_steps": len(target_losses),
            "target_initial_loss": target_losses[0],
            "target_final_loss": target_losses[-1],
            "setup_checks": dict(sorted(setup_checks.items())),
            "wall_seconds": training_seconds,
        },
        "diff": {
            "prompts_evaluated": diff.prompts_evaluated,
            "tokens_processed": diff.tokens_processed,
            "witness_count": len(diff.witnesses),
            "minimized_witness_count": sum(
                item.original_input != item.minimized_input for item in diff.witnesses
            ),
            "cluster_count": len(clusters),
            "selected_cluster": selected_cluster.cluster_id,
            "selected_witness_count": len(selected),
            "nonselected_witness_count": len(nonselected),
            "wall_seconds": diff_seconds,
            "scope": "executed_probe_space",
        },
        "extraction": {
            "attempts": len(extraction_attempts),
            "initial_validation_passed": extraction_attempts[0].validation_passed,
            "initial_selected_teacher_kl": extraction_attempts[0].selected_teacher_kl,
            "initial_nonselected_base_kl": extraction_attempts[0].nonselected_base_kl,
            "compiler_status": minimized.status.value,
            "active_modules": list(minimized.active_modules),
            "total_rank": sum(minimized.ranks.values()),
            "best_step": minimized.best_step,
            "wall_seconds": compile_seconds,
        },
        "cegis": cegis_evidence,
        "minimization": {
            **minimization_evidence,
            "wall_seconds": minimization_seconds,
        },
        "verification": {
            "outcome": verification_report.outcome.value,
            "holdout_outcome": verification_report.holdout_outcome.value,
            "free_generation_records": len(verification_report.free_generation_records),
            "prompt_failures": len(verification_report.prompt_failures),
            "holdout_accesses": len(holdout_gate.access_records),
            "holdout_target_accesses": sum(
                item.role.value == "targets" for item in holdout_gate.access_records
            ),
            "holdout_guard_accesses": sum(
                item.role.value == "guards" for item in holdout_gate.access_records
            ),
            "holdout_target_assertions": len(verification_report.holdout_target_results),
            "holdout_guard_assertions": len(verification_report.holdout_guard_results),
            "sealed_guard_probe_count": len(_SEALED_GUARDS),
            "holdout_opened_after_patch_id": bool(
                initial_bundle.manifest.patch_id
                and holdout_gate.access_records
                and all(
                    item.candidate_id == initial_bundle.manifest.patch_id
                    for item in holdout_gate.access_records
                )
            ),
            "selected_transfer_rate": selected_transfer,
            "unselected_change_rejection_rate": unselected_preservation,
            "unchanged_control_preservation_rate": control_preservation,
            "worst_prompt_base_kl": worst_guard_kl,
            "outputs": output_rows,
        },
        "patch": {
            "patch_id": complete_bundle.manifest.patch_id,
            "factor_tensor_bytes": sum(
                value.numel() * value.element_size() for value in complete_bundle.tensors.values()
            ),
            "bundle_bytes": sum(
                path.stat().st_size for path in complete_bundle.path.rglob("*") if path.is_file()
            ),
            "runtime_unmount_exact": unmount_exact,
            "complete_bundle": True,
            "certificate_hash": certificate.certificate_hash,
        },
        "evidence_scope": {
            "contract_coverage": ["FINITE_PARTIAL", "SEARCH_AUDITED", "SEALED_HOLDOUT"],
            "reproducibility": "DETERMINISTIC_WITHIN_ENVIRONMENT",
            "claim": (
                "Verified under the declared contracts, probe spaces, generation policy, "
                "environment, and search budget."
            ),
        },
        "negative_findings": negative_findings,
        "wall_seconds": time.perf_counter() - started,
    }
    _persist_result(root, final_result)
    return final_result


def run_forkbench(
    output: str | Path | None = None,
    *,
    config: ForkBenchConfig = DEFAULT_FORKBENCH_CONFIG,
) -> dict[str, object]:
    """Run ForkBench and optionally retain its independently inspectable artifacts.

    A temporary artifact tree is still used when ``output`` is omitted; it is
    removed only after all bundle, certificate, mount, and holdout checks have
    completed.
    """

    if output is not None:
        target = Path(output)
        if target.exists():
            raise FileExistsError(target)
        return _run_forkbench_at(target, config)
    with tempfile.TemporaryDirectory(prefix="modelpact-forkbench-") as temporary:
        return _run_forkbench_at(Path(temporary) / "run", config)


__all__ = ["ForkBenchConfig", "run_forkbench"]
