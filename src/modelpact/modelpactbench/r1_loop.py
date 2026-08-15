"""One coherent, executed CPU loop for the ModelPact R1 research claim.

This benchmark deliberately uses the same trained ``TinyCausalLM`` base for
behavioral diff, selective extraction, independent compilation, composition,
semantic merge, semantic rebase, stack resolution, and logical reversion.  The
only acceptance oracle is local model execution under finite data-only behavior
contracts.  No stage returns a preselected outcome.
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from modelpact.adapters.base import ModelBatch
from modelpact.adapters.tiny_lm import (
    TinyCausalLM,
    TinyModelAdapter,
    TinyTrainingConfig,
    save_tiny_checkpoint,
    train_tiny_causal_lm,
)
from modelpact.checkpoints.safetensors import tensor_content_hash
from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
from modelpact.compiler.extract import apply_dense_deltas, extract_behavior_cluster
from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
from modelpact.compiler.package import compilation_delta_program
from modelpact.compiler.result import CompilationResult
from modelpact.compose.closure import (
    CompositionExecutor,
    ContractMargin,
    MarginKind,
    PatchOperand,
    verify_contract_closure,
)
from modelpact.compose.closure import (
    VerificationReport as CompositionVerificationReport,
)
from modelpact.compose.merge import (
    JointCompilationResult,
    MergeBudget,
    MergeDisposition,
    SemanticMergeRequest,
    semantic_merge,
)
from modelpact.compose.stack import (
    PatchReference,
    StackResolutionExecution,
    StackResolutionKind,
    StackResolutionRequest,
    resolve_stack,
)
from modelpact.contracts.ast import (
    AssertionType,
    BehaviorContract,
    CompileObjective,
    GenerationMode,
    GenerationPolicy,
    HoldoutPolicy,
    ModelRequirements,
    ObjectiveType,
    StatisticsPolicy,
    VerificationAssertion,
)
from modelpact.diff.cluster import deterministic_agglomerative
from modelpact.diff.engine import DiffConfig, find_difference_witnesses

# These deterministic fixtures are shared with ForkBench.  They remain inside
# the benchmark package and are never referenced by production compiler code.
from modelpact.modelpactbench.forkbench import (
    _BASE_CORPUS,
    ForkBenchConfig,
    _generated_token,
    _tiny_models,
)
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.patch.mount import mount_patch
from modelpact.rebase.compile import (
    BehavioralRecompileRequest,
    BehavioralRecompileResult,
    RebaseBudget,
    RebaseDisposition,
    RebaseRequest,
    TeacherContext,
    semantic_rebase,
)
from modelpact.rebase.direct import (
    BaseModelDescriptor,
    RebasePatch,
    RebaseVerification,
)
from modelpact.status import (
    CompositionClaim,
    RebaseClaim,
    ReversionGrade,
    VerificationOutcome,
)
from modelpact.util.atomic import atomic_write_bytes, atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical
from modelpact.verify.engine import ExecutionIdentity, VerificationReport, verify_contract
from modelpact.verify.provider import ModelBackedRecordProvider

_DIFF_PROMPTS = ("F:a", "P", "L", "T:x", "C:q", "F:b")
_FACT_PROMPTS = ("F:a",)
_FORMAT_PROMPTS = ("P",)
_LEGACY_GUARDS = ("L", "T:x", "C:q")
_COMPILER_GUARDS = ("F:b", *_LEGACY_GUARDS)
_REBASE_IMPROVEMENT = ("F:b", "Fact:b", "Ask F:b")


@dataclass(frozen=True, slots=True)
class R1LoopConfig:
    """Deterministic CPU budgets for the unified integration."""

    base_steps: int = 160
    target_steps: int = 120
    patch_steps: int = 100
    merge_steps: int = 100
    base_v2_steps: int = 120
    rebase_steps: int = 120

    def __post_init__(self) -> None:
        values = (
            self.base_steps,
            self.target_steps,
            self.patch_steps,
            self.merge_steps,
            self.base_v2_steps,
            self.rebase_steps,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("R1 loop budgets must be positive integers")


DEFAULT_R1_LOOP_CONFIG = R1LoopConfig()


@dataclass(frozen=True, slots=True)
class _TeacherExample:
    batch: ModelBatch
    logits: Tensor


@dataclass(frozen=True, slots=True)
class _ExecutableContract:
    contract: BehaviorContract
    reference_model: nn.Module


def _state_digest(model: nn.Module) -> str:
    return hash_canonical(
        {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "hash": tensor_content_hash(value),
            }
            for name, value in sorted(model.state_dict().items())
        }
    )


def _delta_digest(delta: Mapping[str, Tensor]) -> str:
    return hash_canonical(
        {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "hash": tensor_content_hash(value),
            }
            for name, value in sorted(delta.items())
        }
    )


def _patch_id(
    *,
    base_signature: str,
    contract_hash: str,
    delta: Mapping[str, Tensor],
) -> str:
    return hash_canonical(
        {
            "schema_version": 1,
            "base_signature": base_signature,
            "contract_hash": contract_hash,
            "delta_hash": _delta_digest(delta),
        }
    )


def _teacher_examples(
    adapter: TinyModelAdapter,
    teacher: nn.Module,
    prompts: Sequence[str],
) -> tuple[_TeacherExample, ...]:
    examples: list[_TeacherExample] = []
    for prompt in prompts:
        batch = adapter.tokenizer().batch((prompt,))
        with torch.no_grad():
            logits = adapter.forward_logits(teacher, batch).detach().cpu()
        examples.append(_TeacherExample(batch, logits))
    return tuple(examples)


def _teacher_kl(adapter: TinyModelAdapter) -> Callable[[nn.Module, _TeacherExample], Tensor]:
    def loss(model: nn.Module, example: _TeacherExample) -> Tensor:
        student_logits = adapter.forward_logits(model, example.batch)
        positions = example.batch.attention_mask.to(student_logits.device).sum(dim=1) - 1
        rows = torch.arange(student_logits.shape[0], device=student_logits.device)
        student = student_logits[rows, positions].to(torch.float64)
        teacher = example.logits.to(student_logits.device)[rows, positions].to(torch.float64)
        teacher_probability = torch.softmax(teacher, dim=-1).clamp_min(1e-12)
        student_log_probability = torch.log_softmax(student, dim=-1)
        return (
            (teacher_probability * (teacher_probability.log() - student_log_probability))
            .sum(dim=-1)
            .mean()
        )

    return loss


def _objective(
    adapter: TinyModelAdapter,
    teacher: nn.Module,
    prompts: Sequence[str],
    *,
    objective_id: str,
) -> DifferentiableObjective:
    return DifferentiableObjective(
        objective_id,
        _teacher_examples(adapter, teacher, prompts),
        _teacher_kl(adapter),
    )


def _guard(
    adapter: TinyModelAdapter,
    teacher: nn.Module,
    prompts: Sequence[str],
    *,
    constraint_id: str,
    maximum: float = 0.1,
) -> DifferentiableConstraint:
    return DifferentiableConstraint(
        constraint_id,
        _teacher_examples(adapter, teacher, prompts),
        _teacher_kl(adapter),
        maximum=maximum,
    )


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return ("".join(canonical_dumps(dict(row)) + "\n" for row in rows)).encode("utf-8")


def _write_probe_workspace(
    root: Path,
    adapter: TinyModelAdapter,
    base: nn.Module,
    target: nn.Module,
) -> Path:
    probes = root / "contracts"
    rows: dict[str, bytes] = {
        "fact-compile.jsonl": _jsonl(({"id": "fact-compile", "prompt": "F:a"},)),
        "fact-target.jsonl": _jsonl(
            (
                {
                    "id": "fact-target",
                    "prompt": "F:a",
                    "expected": _generated_token(adapter, target, "F:a")[1],
                },
            )
        ),
        "format-compile.jsonl": _jsonl(({"id": "format-compile", "prompt": "P"},)),
        "format-target.jsonl": _jsonl(
            (
                {
                    "id": "format-target",
                    "prompt": "P",
                    "expected": _generated_token(adapter, target, "P")[1],
                },
            )
        ),
        "legacy-guards.jsonl": _jsonl(
            tuple(
                {
                    "id": f"legacy-{index}",
                    "prompt": prompt,
                    "expected": _generated_token(adapter, base, prompt)[1],
                }
                for index, prompt in enumerate(_LEGACY_GUARDS)
            )
        ),
    }
    for name, content in sorted(rows.items()):
        atomic_write_bytes(probes / name, content, overwrite=False)
    return probes


def _behavior_contract(
    manifest: ModelManifest,
    *,
    contract_id: str,
    target_source: str,
    compile_source: str,
) -> BehaviorContract:
    return BehaviorContract(
        schema_version=1,
        id=contract_id,
        contract_version=1,
        model_requirements=ModelRequirements(
            tokenizer_hash=manifest.signature.tokenizer_hash,
            base_signature=manifest.signature.signature_hash,
            architecture_hash=manifest.signature.architecture_hash,
            state_schema_hash=manifest.signature.state_schema_hash,
            adapter_id=manifest.signature.adapter_id,
        ),
        objectives=(
            CompileObjective(
                f"{contract_id}-teacher-kl",
                ObjectiveType.TEACHER_KL,
                compile_source,
            ),
        ),
        targets=(
            VerificationAssertion(
                f"{contract_id}-free-generation",
                AssertionType.FREE_GENERATION_MATCH,
                target_source,
                {"minimum_pass_rate": 1.0},
            ),
        ),
        guards=(
            VerificationAssertion(
                f"{contract_id}-preserve-generation",
                AssertionType.FREE_GENERATION_MATCH,
                "legacy-guards.jsonl",
                {"minimum_pass_rate": 1.0},
            ),
        ),
        holdout=HoldoutPolicy(sealed=True),
        statistics=StatisticsPolicy(
            confidence_level=0.95,
            bootstrap_samples=64,
            bootstrap_seed=731,
        ),
        generation=GenerationPolicy(
            mode=GenerationMode.GREEDY,
            max_new_tokens=1,
            seeds=(0,),
        ),
        description=(
            "Finite ModelPactBench contract. It claims only exact one-token generation "
            "on the committed target and legacy-control probes."
        ),
    )


def _execution_identity(manifest: ModelManifest) -> ExecutionIdentity:
    return ExecutionIdentity(
        adapter_id=manifest.signature.adapter_id,
        base_signature=manifest.signature.signature_hash,
        tokenizer_hash=manifest.signature.tokenizer_hash,
        architecture_hash=manifest.signature.architecture_hash,
        state_schema_hash=manifest.signature.state_schema_hash,
    )


def _execute_behavior_contract(
    executable: _ExecutableContract,
    *,
    adapter: TinyModelAdapter,
    candidate: nn.Module,
    base: nn.Module,
    probe_root: Path,
    identity: ExecutionIdentity,
) -> VerificationReport:
    provider = ModelBackedRecordProvider(
        adapter=adapter,
        model=candidate,
        base_model=base,
        reference_model=executable.reference_model,
        contract_root=probe_root,
        generation_policy=executable.contract.generation,
    )
    return verify_contract(
        executable.contract,
        identity=identity,
        provider=provider,
    )


def _signed_report_margin(report: VerificationReport) -> float:
    results = (*report.target_results, *report.guard_results)
    if report.outcome is not VerificationOutcome.PASS:
        failing = [item.margin for item in results if item.margin is not None and item.margin < 0]
        return min(failing, default=-1.0)
    margins = [item.margin for item in results if item.margin is not None]
    return min(margins, default=0.0)


def _composition_executor(
    *,
    adapter: TinyModelAdapter,
    base: nn.Module,
    contracts: Mapping[str, _ExecutableContract],
    probe_root: Path,
    identity: ExecutionIdentity,
) -> CompositionExecutor:
    def execute(
        delta: Mapping[str, Tensor], contract_ids: tuple[str, ...]
    ) -> CompositionVerificationReport:
        candidate = apply_dense_deltas(base, dict(delta))
        margins: list[ContractMargin] = []
        outcomes: list[VerificationOutcome] = []
        for contract_id in contract_ids:
            executable = contracts[contract_id]
            report = _execute_behavior_contract(
                executable,
                adapter=adapter,
                candidate=candidate,
                base=base,
                probe_root=probe_root,
                identity=identity,
            )
            outcomes.append(report.outcome)
            margins.append(
                ContractMargin(
                    contract_id,
                    MarginKind.FREE_GENERATION,
                    _signed_report_margin(report),
                    {
                        "verification_outcome": report.outcome.value,
                        "verification_result_hash": report.result_hash,
                        "prompt_failures": len(report.prompt_failures),
                    },
                )
            )
        outcome = (
            VerificationOutcome.PASS
            if outcomes and all(item is VerificationOutcome.PASS for item in outcomes)
            else VerificationOutcome.FAIL
        )
        return CompositionVerificationReport(outcome, tuple(margins))

    return execute


def _outputs(
    adapter: TinyModelAdapter,
    model: nn.Module,
    prompts: Sequence[str],
) -> dict[str, str]:
    return {prompt: _generated_token(adapter, model, prompt)[1] for prompt in prompts}


def _teacher_margin(
    adapter: TinyModelAdapter,
    candidate: nn.Module,
    teacher: nn.Module,
    prompts: Sequence[str],
) -> tuple[float, bool]:
    margins: list[float] = []
    exact = True
    for prompt in prompts:
        batch = adapter.tokenizer().batch((prompt,))
        with torch.no_grad():
            candidate_logits = adapter.forward_logits(candidate, batch)[0, -1].float()
            teacher_logits = adapter.forward_logits(teacher, batch)[0, -1].float()
        expected_token = int(torch.argmax(teacher_logits).item())
        expected = candidate_logits[expected_token]
        alternatives = candidate_logits.clone()
        alternatives[expected_token] = torch.finfo(alternatives.dtype).min
        margins.append(float((expected - alternatives.max()).detach().cpu()))
        exact = exact and (
            _generated_token(adapter, candidate, prompt)[0]
            == _generated_token(adapter, teacher, prompt)[0]
        )
    margin = min(margins)
    if not exact:
        margin = min(margin, -1e-6)
    return margin, exact


def _descriptor(manifest: ModelManifest, model: nn.Module) -> BaseModelDescriptor:
    shapes = {
        name: tuple(parameter.shape)
        for name, parameter in sorted(model.named_parameters(remove_duplicate=False))
    }
    return BaseModelDescriptor(
        signature=manifest.signature.signature_hash,
        architecture_id=manifest.signature.architecture_hash,
        module_schema_hash=manifest.signature.state_schema_hash,
        tokenizer_hash=manifest.signature.tokenizer_hash,
        output_semantics="causal_lm",
        module_shapes=shapes,
        family_id="modelpact-tiny-causal-lm",
    )


def _base_v2_corpus() -> tuple[str, ...]:
    prefixes = ("F:b", "Fact:b", "Ask F:b", "Query F:b")
    return tuple(f"{text[:-1]}D" if text.startswith(prefixes) else text for text in _BASE_CORPUS)


def _compilation_summary(result: CompilationResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "optimization_steps_executed": len(result.evidence),
        "best_step": result.best_step,
        "best_target_loss": result.best_target_loss,
        "active_modules": list(result.active_modules),
        "module_ranks": dict(sorted(result.ranks.items())),
        "total_rank": sum(result.ranks.values()),
        "delta_hash": _delta_digest(result.deltas),
        "primal_dual": result.metadata.get("primal_dual") is True,
    }


def _run_at(root: Path, config: R1LoopConfig) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()

    tokenizer, adapter, base, target, base_losses, target_losses = _tiny_models(
        ForkBenchConfig(
            base_steps=config.base_steps,
            target_steps=config.target_steps,
            compiler_steps=config.patch_steps,
            cegis_rounds=1,
            search_budget_per_round=1,
            minimization_budget=1,
        )
    )
    base_before = _state_digest(base)
    target_before = _state_digest(target)
    base_checkpoint = save_tiny_checkpoint(base, checkpoints / "base", tokenizer=tokenizer)
    target_checkpoint = save_tiny_checkpoint(target, checkpoints / "target", tokenizer=tokenizer)
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

    setup_outputs = {
        "base": _outputs(adapter, base, _DIFF_PROMPTS),
        "target": _outputs(adapter, target, _DIFF_PROMPTS),
    }
    expected_setup = {
        "F:a": ("R", "G"),
        "P": ("X", "{"),
        "L": ("M", "K"),
        "T:x": ("Y", "Z"),
        "C:q": ("1", "2"),
        "F:b": ("B", "B"),
    }
    setup_passed = all(
        (setup_outputs["base"][prompt], setup_outputs["target"][prompt]) == expected
        for prompt, expected in expected_setup.items()
    )

    difference = find_difference_witnesses(
        adapter,
        base,
        target,
        _DIFF_PROMPTS,
        config=DiffConfig(
            divergence_threshold=0.1,
            search_budget=len(_DIFF_PROMPTS),
            generation_max_new_tokens=1,
            activation_dimensions=2,
            gradient_dimensions=2,
            maximum_activation_points=1,
            maximum_gradient_modules=1,
            seed=17,
        ),
    )
    clusters = deterministic_agglomerative(
        difference.witnesses,
        maximum_clusters=5,
        distance_threshold=1.0,
    )
    by_original = {item.original_input: item for item in difference.witnesses}
    selected = (by_original["F:a"],)
    # The extraction optimizer treats the format witness as nonselected so the
    # first patch demonstrably leaves that learned target change behind.  It is
    # not made a permanent guard in the fact contract: a later explicit format
    # patch is therefore allowed to change this domain.
    nonselected = tuple(by_original[prompt] for prompt in ("P", "L", "T:x", "C:q"))
    selected_cluster = next(
        item.cluster_id for item in clusters if selected[0].witness_id in item.witness_ids
    )

    fact_extraction = extract_behavior_cluster(
        adapter,
        base,
        target,
        selected,
        nonselected,
        additional_guards=("F:b",),
        optimizer_config=OptimizerConfig(
            maximum_rank=2,
            maximum_modules=3,
            steps=config.patch_steps,
            learning_rate=0.05,
            patience=config.patch_steps,
            complexity_weight=1e-7,
            seed=100,
        ),
        maximum_selected_kl=0.2,
        maximum_nonselected_base_kl=0.1,
    )
    fact_result = fact_extraction.compiler_result

    format_result = compile_low_rank_patch(
        base,
        (_objective(adapter, target, _FORMAT_PROMPTS, objective_id="format-teacher-kl"),),
        (
            _guard(
                adapter,
                base,
                (*_COMPILER_GUARDS, "F:a"),
                constraint_id="format-base-guard",
            ),
        ),
        config=OptimizerConfig(
            maximum_rank=1,
            maximum_modules=2,
            steps=config.patch_steps,
            learning_rate=0.05,
            patience=config.patch_steps,
            complexity_weight=1e-7,
            seed=24,
        ),
    )

    probe_root = _write_probe_workspace(root, adapter, base, target)
    fact_contract = _behavior_contract(
        base_manifest,
        contract_id="r1-fact-behavior",
        target_source="fact-target.jsonl",
        compile_source="fact-compile.jsonl",
    )
    format_contract = _behavior_contract(
        base_manifest,
        contract_id="r1-format-behavior",
        target_source="format-target.jsonl",
        compile_source="format-compile.jsonl",
    )
    contracts = {
        fact_contract.id: _ExecutableContract(fact_contract, target),
        format_contract.id: _ExecutableContract(format_contract, target),
    }
    identity = _execution_identity(base_manifest)
    executor = _composition_executor(
        adapter=adapter,
        base=base,
        contracts=contracts,
        probe_root=probe_root,
        identity=identity,
    )

    fact_patch_id = _patch_id(
        base_signature=base_manifest.signature.signature_hash,
        contract_hash=fact_contract.contract_id,
        delta=fact_result.deltas,
    )
    format_patch_id = _patch_id(
        base_signature=base_manifest.signature.signature_hash,
        contract_hash=format_contract.contract_id,
        delta=format_result.deltas,
    )
    fact_operand = PatchOperand(
        patch_id=fact_patch_id,
        base_signature=base_manifest.signature.signature_hash,
        module_schema_hash=base_manifest.signature.state_schema_hash,
        delta=fact_result.deltas,
        contract_ids=(fact_contract.id,),
    )
    format_operand = PatchOperand(
        patch_id=format_patch_id,
        base_signature=base_manifest.signature.signature_hash,
        module_schema_hash=base_manifest.signature.state_schema_hash,
        delta=format_result.deltas,
        contract_ids=(format_contract.id,),
    )
    fact_individual = verify_contract_closure((fact_operand,), executor=executor)
    format_individual = verify_contract_closure((format_operand,), executor=executor)

    runtime = copy.deepcopy(base)
    runtime_before = _state_digest(runtime)
    fact_program, fact_tensors = compilation_delta_program(
        fact_result,
        base_manifest.state_schema,
    )
    mount = mount_patch(
        runtime,
        fact_program,
        fact_tensors,
        state_schema=base_manifest.state_schema,
    )
    try:
        mounted_outputs = _outputs(adapter, runtime, (*_FACT_PROMPTS, *_LEGACY_GUARDS))
    finally:
        mount.unmount()
    unmounted_outputs = _outputs(adapter, runtime, _FACT_PROMPTS)
    runtime_unmount_exact = runtime_before == _state_digest(runtime)

    joint_compilation: CompilationResult | None = None

    def compile_union(request: SemanticMergeRequest) -> JointCompilationResult:
        nonlocal joint_compilation
        joint_compilation = compile_low_rank_patch(
            base,
            (
                _objective(adapter, target, _FACT_PROMPTS, objective_id="merge-fact-kl"),
                _objective(adapter, target, _FORMAT_PROMPTS, objective_id="merge-format-kl"),
            ),
            (
                _guard(
                    adapter,
                    base,
                    _COMPILER_GUARDS,
                    constraint_id="merge-base-guard",
                ),
            ),
            config=OptimizerConfig(
                maximum_rank=4,
                maximum_modules=5,
                steps=request.budget.maximum_steps,
                learning_rate=0.04,
                patience=request.budget.maximum_steps,
                complexity_weight=1e-7,
                seed=333,
            ),
        )
        return JointCompilationResult(
            candidate_delta=joint_compilation.deltas if joint_compilation.feasible else None,
            optimization_succeeded=joint_compilation.feasible,
            budget_exhausted=(
                not joint_compilation.feasible
                and len(joint_compilation.evidence) >= request.budget.maximum_steps
            ),
            steps_executed=len(joint_compilation.evidence),
            restarts_executed=1,
            violated_contracts=tuple(sorted(joint_compilation.violated_constraints)),
            diagnostics={
                "optimizer": joint_compilation.metadata.get("optimizer"),
                "primal_dual": joint_compilation.metadata.get("primal_dual"),
                "active_modules": list(joint_compilation.active_modules),
                "total_rank": sum(joint_compilation.ranks.values()),
                "parent_sum_received": bool(request.initial_delta),
                "initializer": "contrastive_gradient",
            },
            failure_reason=(
                None
                if joint_compilation.feasible
                else "no feasible union-contract candidate found within the declared budget"
            ),
        )

    merged = semantic_merge(
        (fact_operand, format_operand),
        executor=executor,
        compiler=compile_union,
        budget=MergeBudget(maximum_steps=config.merge_steps),
    )
    merged_model = apply_dense_deltas(base, dict(merged.delta))

    fact_patched = apply_dense_deltas(base, fact_result.deltas)
    base_v2 = copy.deepcopy(base)
    base_v2_losses = train_tiny_causal_lm(
        base_v2,
        _base_v2_corpus(),
        tokenizer=tokenizer,
        config=TinyTrainingConfig(
            steps=config.base_v2_steps,
            batch_size=24,
            learning_rate=0.01,
            seed=45,
        ),
    )
    adapter.prepare(base_v2)
    base_v2_checkpoint = save_tiny_checkpoint(
        base_v2,
        checkpoints / "base-v2",
        tokenizer=tokenizer,
    )
    base_v2_manifest = build_model_manifest(
        base_v2,
        checkpoint=base_v2_checkpoint,
        adapter_id=adapter.adapter_id,
        architecture_config=base_v2.config.to_dict(),
    )
    base_v2_before = _state_digest(base_v2)
    source_descriptor = _descriptor(base_manifest, base)
    target_descriptor = _descriptor(base_v2_manifest, base_v2)
    rebase_verifications: list[dict[str, object]] = []
    rebase_compilation: CompilationResult | None = None

    def apply_to_v2(delta: Mapping[str, Tensor], target: BaseModelDescriptor) -> object:
        if target.signature != target_descriptor.signature:
            raise ValueError("R1 loop applier received an unexpected target base")
        return apply_dense_deltas(base_v2, dict(delta))

    def verify_rebase(
        candidate: object,
        target_contract_ids: tuple[str, ...],
        guard_contract_ids: tuple[str, ...],
    ) -> RebaseVerification:
        if not isinstance(candidate, TinyCausalLM):
            raise TypeError("R1 loop rebase verifier requires TinyCausalLM")
        target_margin, target_exact = _teacher_margin(
            adapter,
            candidate,
            fact_patched,
            _FACT_PROMPTS,
        )
        target_margins = dict.fromkeys(target_contract_ids, target_margin)
        guard_margins: dict[str, float] = {}
        guard_exact: dict[str, bool] = {}
        for contract_id in guard_contract_ids:
            prompts: Sequence[str]
            if contract_id == "new-base-improvement":
                prompts = _REBASE_IMPROVEMENT
            elif contract_id == "new-base-other-controls":
                prompts = _FORMAT_PROMPTS
            else:
                prompts = _LEGACY_GUARDS
            margin, exact = _teacher_margin(adapter, candidate, base_v2, prompts)
            guard_margins[contract_id] = margin
            guard_exact[contract_id] = exact
        passed = target_exact and all(guard_exact.values())
        outcome = VerificationOutcome.PASS if passed else VerificationOutcome.FAIL
        rebase_verifications.append(
            {
                "outcome": outcome.value,
                "target_exact": target_exact,
                "guard_exact": dict(sorted(guard_exact.items())),
                "outputs": _outputs(
                    adapter,
                    candidate,
                    (*_FACT_PROMPTS, *_REBASE_IMPROVEMENT, *_LEGACY_GUARDS, "P"),
                ),
            }
        )
        return RebaseVerification(
            outcome,
            target_margins=target_margins,
            guard_margins=guard_margins,
            prompt_failures=(() if passed else ("finite generation contract failed",)),
        )

    def build_teachers(request: RebaseRequest) -> TeacherContext:
        del request
        margin, _ = _teacher_margin(adapter, fact_patched, target, _FACT_PROMPTS)
        return TeacherContext(
            old_patched_teacher=fact_patched,
            new_unpatched_teacher=base_v2,
            old_behavior_margins={fact_contract.id: margin},
            evidence_count=len(_FACT_PROMPTS),
        )

    def recompile_on_v2(request: BehavioralRecompileRequest) -> BehavioralRecompileResult:
        nonlocal rebase_compilation
        old_teacher = cast(nn.Module, request.old_patched_teacher)
        new_teacher = cast(nn.Module, request.new_unpatched_teacher)
        rebase_compilation = compile_low_rank_patch(
            base_v2,
            (
                _objective(
                    adapter,
                    old_teacher,
                    _FACT_PROMPTS,
                    objective_id="rebase-old-patched-teacher-kl",
                ),
            ),
            (
                _guard(
                    adapter,
                    new_teacher,
                    (*_REBASE_IMPROVEMENT, *_LEGACY_GUARDS, "P"),
                    constraint_id="rebase-new-base-guard",
                ),
            ),
            config=OptimizerConfig(
                maximum_rank=4,
                maximum_modules=5,
                steps=request.budget.maximum_steps,
                learning_rate=0.04,
                patience=request.budget.maximum_steps,
                complexity_weight=1e-7,
                seed=444,
            ),
        )
        return BehavioralRecompileResult(
            candidate_delta=rebase_compilation.deltas if rebase_compilation.feasible else None,
            optimization_succeeded=rebase_compilation.feasible,
            budget_exhausted=(
                not rebase_compilation.feasible
                and len(rebase_compilation.evidence) >= request.budget.maximum_steps
            ),
            steps_executed=len(rebase_compilation.evidence),
            restarts_executed=1,
            violated_contracts=tuple(sorted(rebase_compilation.violated_constraints)),
            complexity={
                "active_modules": len(rebase_compilation.active_modules),
                "total_rank": sum(rebase_compilation.ranks.values()),
                "parameters": sum(
                    left.numel() + right.numel()
                    for left, right in rebase_compilation.factors.values()
                ),
            },
            failure_reason=(
                None
                if rebase_compilation.feasible
                else "no feasible rebase candidate found within the declared budget"
            ),
        )

    rebased = semantic_rebase(
        RebaseRequest(
            patch=RebasePatch(
                patch_id=fact_patch_id,
                source_base_signature=source_descriptor.signature,
                delta=fact_result.deltas,
                target_contract_ids=(fact_contract.id,),
                preservation_contract_ids=("legacy-controls",),
            ),
            source_base=source_descriptor,
            target_base=target_descriptor,
            new_base_guard_ids=("new-base-improvement", "new-base-other-controls"),
            budget=RebaseBudget(config.rebase_steps),
        ),
        applier=apply_to_v2,
        verifier=verify_rebase,
        teacher_builder=build_teachers,
        recompiler=recompile_on_v2,
    )

    policy_hash = hash_canonical(
        {
            "generation": fact_contract.generation.to_dict(),
            "statistics": fact_contract.statistics.to_dict(),
        }
    )
    fact_reference = PatchReference(
        patch_id=fact_patch_id,
        patch_hash=fact_patch_id,
        base_hash=base_manifest.signature.signature_hash,
        contract_hashes=(fact_contract.contract_id,),
        artifact_hash=_delta_digest(fact_result.deltas),
    )
    format_reference = PatchReference(
        patch_id=format_patch_id,
        patch_hash=format_patch_id,
        base_hash=base_manifest.signature.signature_hash,
        contract_hashes=(format_contract.contract_id,),
        artifact_hash=_delta_digest(format_result.deltas),
    )

    def resolve_merged(request: StackResolutionRequest) -> StackResolutionExecution:
        report = executor(merged.delta, tuple(sorted(contracts)))
        if report.outcome is not VerificationOutcome.PASS:
            return StackResolutionExecution(
                StackResolutionKind.EMPIRICAL_FAILURE,
                None,
                policy_hash,
                hash_canonical({"contracts": sorted(contracts)}),
                warnings=("merged stack failed re-executed union contracts",),
            )
        return StackResolutionExecution(
            StackResolutionKind.VERIFIED_COMPOSITE_PATCH,
            _delta_digest(merged.delta),
            policy_hash,
            hash_canonical({"contracts": sorted(contracts)}),
            certificate_hash=hash_canonical(
                {
                    "base": request.base_hash,
                    "patches": [item.patch_id for item in request.patches],
                    "margins": {item.contract_id: item.margin for item in report.margins},
                }
            ),
        )

    resolved_stack = resolve_stack(
        base_hash=base_manifest.signature.signature_hash,
        patches=(format_reference, fact_reference),
        resolver=resolve_merged,
        repair_conflicts=True,
        subset_audit_budget=3,
    )

    reverted_verification: CompositionVerificationReport | None = None

    def resolve_remaining(request: StackResolutionRequest) -> StackResolutionExecution:
        nonlocal reverted_verification
        reverted_verification = executor(fact_result.deltas, (fact_contract.id,))
        if reverted_verification.outcome is not VerificationOutcome.PASS:
            return StackResolutionExecution(
                StackResolutionKind.EMPIRICAL_FAILURE,
                None,
                policy_hash,
                fact_contract.contract_id,
                warnings=("remaining fact contract failed after logical removal",),
            )
        return StackResolutionExecution(
            StackResolutionKind.NAIVE_ADDITIVE_STACK,
            _delta_digest(fact_result.deltas),
            policy_hash,
            fact_contract.contract_id,
            certificate_hash=hash_canonical(
                {
                    "base": request.base_hash,
                    "remaining": [item.patch_id for item in request.patches],
                    "margin": reverted_verification.margins[0].margin,
                }
            ),
        )

    reverted_stack = resolve_stack(
        base_hash=base_manifest.signature.signature_hash,
        patches=(fact_reference,),
        resolver=resolve_remaining,
        repair_conflicts=False,
        subset_audit_budget=1,
    )
    reverted_model = apply_dense_deltas(base, fact_result.deltas)

    source_integrity = {
        "base_unchanged": base_before == _state_digest(base),
        "target_unchanged": target_before == _state_digest(target),
        "base_v2_unchanged_after_rebase": base_v2_before == _state_digest(base_v2),
    }
    individual_pass = (
        fact_individual.claim is CompositionClaim.COMPOSITION_CLOSED
        and format_individual.claim is CompositionClaim.COMPOSITION_CLOSED
    )
    merged_pass = (
        merged.disposition is MergeDisposition.SEMANTIC_MERGE_VERIFIED
        and merged.compiler_invoked
        and merged.compilation is not None
        and merged.compilation.steps_executed > 0
    )
    rebase_pass = (
        rebased.claim is RebaseClaim.SEMANTIC_REBASE_VERIFIED
        and rebased.disposition is RebaseDisposition.SEMANTIC_REBASE_VERIFIED
        and rebased.direct_transfer.attempted
        and not rebased.direct_transfer.verified
        and rebased.recompile is not None
        and rebased.recompile.steps_executed > 0
    )
    revert_pass = (
        reverted_verification is not None
        and reverted_verification.outcome is VerificationOutcome.PASS
        and _generated_token(adapter, reverted_model, "F:a")[1] == "G"
        and _generated_token(adapter, reverted_model, "P")[1] == "X"
    )
    success = bool(
        setup_passed
        and len(difference.witnesses) >= 5
        and fact_extraction.validation_passed
        and fact_result.feasible
        and format_result.feasible
        and individual_pass
        and runtime_unmount_exact
        and merged.naive_composition.claim is CompositionClaim.SEMANTIC_CONFLICT
        and merged_pass
        and rebase_pass
        and resolved_stack.execution.kind is StackResolutionKind.VERIFIED_COMPOSITE_PATCH
        and revert_pass
        and all(source_integrity.values())
    )

    if joint_compilation is None or rebase_compilation is None:
        raise RuntimeError(
            "the merge and rebase optimization callbacks were not both executed: "
            f"merge={joint_compilation is not None}, rebase={rebase_compilation is not None}, "
            f"naive={merged.naive_composition.claim.value}, direct_rebase="
            f"{rebased.direct_transfer.verified}"
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": "R1 Unified Tiny Loop",
        "status": "PASS" if success else "FAIL",
        "success": success,
        "resource_policy": {
            "device": "cpu",
            "base_steps": config.base_steps,
            "target_steps": config.target_steps,
            "patch_steps": config.patch_steps,
            "merge_steps": config.merge_steps,
            "base_v2_steps": config.base_v2_steps,
            "rebase_steps": config.rebase_steps,
        },
        "model": {
            "adapter_id": adapter.adapter_id,
            "architecture": "TinyCausalLM",
            "base_signature": base_manifest.signature.signature_hash,
            "target_signature": target_manifest.signature.signature_hash,
            "base_v2_signature": base_v2_manifest.signature.signature_hash,
            "base_training": {
                "steps": len(base_losses),
                "initial_loss": base_losses[0],
                "final_loss": base_losses[-1],
            },
            "target_training": {
                "steps": len(target_losses),
                "initial_loss": target_losses[0],
                "final_loss": target_losses[-1],
            },
            "base_v2_training": {
                "steps": len(base_v2_losses),
                "initial_loss": base_v2_losses[0],
                "final_loss": base_v2_losses[-1],
            },
            "setup_outputs": setup_outputs,
        },
        "behavioral_diff": {
            "prompts_evaluated": difference.prompts_evaluated,
            "witness_count": len(difference.witnesses),
            "cluster_count": len(clusters),
            "selected_cluster": selected_cluster,
            "selected_witness_id": selected[0].witness_id,
            "nonselected_witness_ids": [item.witness_id for item in nonselected],
            "scope": "finite_executed_prompt_space",
        },
        "first_patch_extraction": {
            "patch_id": fact_patch_id,
            "validation_passed": fact_extraction.validation_passed,
            "selected_teacher_kl": fact_extraction.selected_teacher_kl,
            "nonselected_base_kl": fact_extraction.nonselected_base_kl,
            "contract_hash": fact_contract.contract_id,
            "contract_outcome": (
                fact_individual.verification.outcome.value
                if fact_individual.verification is not None
                else VerificationOutcome.INCONCLUSIVE.value
            ),
            "outputs": _outputs(
                adapter,
                fact_patched,
                (*_FACT_PROMPTS, *_FORMAT_PROMPTS, *_COMPILER_GUARDS),
            ),
            **_compilation_summary(fact_result),
        },
        "second_patch_compilation": {
            "patch_id": format_patch_id,
            "contract_hash": format_contract.contract_id,
            "contract_outcome": (
                format_individual.verification.outcome.value
                if format_individual.verification is not None
                else VerificationOutcome.INCONCLUSIVE.value
            ),
            "outputs": _outputs(
                adapter,
                apply_dense_deltas(base, format_result.deltas),
                (*_FACT_PROMPTS, *_FORMAT_PROMPTS, *_COMPILER_GUARDS),
            ),
            **_compilation_summary(format_result),
        },
        "runtime_mount": {
            "mounted_outputs": mounted_outputs,
            "unmounted_outputs": unmounted_outputs,
            "unmount_exact": runtime_unmount_exact,
            "base_source_unchanged": source_integrity["base_unchanged"],
        },
        "composition": {
            "individual_contracts_pass": individual_pass,
            "naive_claim": merged.naive_composition.claim.value,
            "naive_outputs": _outputs(
                adapter,
                apply_dense_deltas(base, dict(merged.naive_composition.resolved_delta)),
                (*_FACT_PROMPTS, *_FORMAT_PROMPTS, *_COMPILER_GUARDS),
            ),
            "failed_contracts": (
                []
                if merged.naive_composition.verification is None
                else [
                    item.contract_id
                    for item in merged.naive_composition.verification.margins
                    if not item.passed
                ]
            ),
        },
        "semantic_merge": {
            "disposition": merged.disposition.value,
            "claim": merged.claim.value,
            "compiler_invoked": merged.compiler_invoked,
            "optimization_steps_executed": len(joint_compilation.evidence),
            "fresh_delta_hash": _delta_digest(merged.delta),
            "parent_sum_delta_hash": _delta_digest(merged.naive_composition.resolved_delta),
            "fresh_delta_differs_from_parent_sum": (
                _delta_digest(merged.delta)
                != _delta_digest(merged.naive_composition.resolved_delta)
            ),
            "verification_outcome": (
                merged.verification.outcome.value
                if merged.verification is not None
                else VerificationOutcome.INCONCLUSIVE.value
            ),
            "outputs": _outputs(
                adapter,
                merged_model,
                (*_FACT_PROMPTS, *_FORMAT_PROMPTS, *_COMPILER_GUARDS),
            ),
            **_compilation_summary(joint_compilation),
        },
        "semantic_rebase": {
            "claim": rebased.claim.value,
            "disposition": rebased.disposition.value,
            "direct_attempted": rebased.direct_transfer.attempted,
            "direct_verified": rebased.direct_transfer.verified,
            "direct_outcome": (
                rebased.direct_transfer.verification.outcome.value
                if rebased.direct_transfer.verification is not None
                else VerificationOutcome.NOT_APPLICABLE.value
            ),
            "direct_outputs": rebase_verifications[0]["outputs"],
            "recompile_optimization_steps": len(rebase_compilation.evidence),
            "final_outputs": rebase_verifications[-1]["outputs"],
            "base_v2_unpatched_outputs": _outputs(
                adapter,
                base_v2,
                (*_FACT_PROMPTS, *_REBASE_IMPROVEMENT, *_LEGACY_GUARDS, "P"),
            ),
            "new_base_improvement": "F:b->D",
            **_compilation_summary(rebase_compilation),
        },
        "stack_and_revert": {
            "resolved_kind": resolved_stack.execution.kind.value,
            "resolved_artifact_hash": resolved_stack.lock.resolved_artifact_hash,
            "dependency_order": list(resolved_stack.dependency_order),
            "lock": resolved_stack.lock.to_dict(),
            "removed_patch_id": format_patch_id,
            "reversion_grade": ReversionGrade.VERIFIED_LOGICAL_STACK_RECONSTRUCTED.value,
            "remaining_resolution": reverted_stack.execution.kind.value,
            "remaining_contract_outcome": (
                reverted_verification.outcome.value
                if reverted_verification is not None
                else VerificationOutcome.INCONCLUSIVE.value
            ),
            "remaining_outputs": _outputs(
                adapter,
                reverted_model,
                (*_FACT_PROMPTS, *_FORMAT_PROMPTS, *_LEGACY_GUARDS),
            ),
            "numeric_subtraction_used": False,
        },
        "source_integrity": source_integrity,
        "evidence_scope": {
            "contract_coverage": "FINITE_PARTIAL",
            "composition_coverage": "executed_declared_two_patch_stack",
            "reproducibility": "DETERMINISTIC_WITHIN_ENVIRONMENT",
            "claim": (
                "Verified under the declared finite contracts, probe space, deterministic "
                "generation policy, local environment, and optimization budgets."
            ),
            "unsupported_claims": [
                "GLOBAL_MINIMUM",
                "SEALED_HOLDOUT_VERIFIED",
                "UNIVERSAL_BEHAVIOR_PRESERVATION",
            ],
        },
    }
    atomic_write_text(root / "result.json", canonical_dumps(result) + "\n", overwrite=False)
    return result


def run_r1_loop(
    output: str | Path | None = None,
    *,
    config: R1LoopConfig = DEFAULT_R1_LOOP_CONFIG,
) -> dict[str, object]:
    """Execute the complete bounded R1 loop and return canonicalizable evidence."""

    if output is not None:
        target = Path(output)
        if target.exists():
            raise FileExistsError(target)
        return _run_at(target, config)
    with tempfile.TemporaryDirectory(prefix="modelpact-r1-loop-") as temporary:
        return _run_at(Path(temporary) / "run", config)


__all__ = ["R1LoopConfig", "run_r1_loop"]
