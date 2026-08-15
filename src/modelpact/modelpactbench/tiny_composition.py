"""Executed TinyCausalLM composition ground truth and benign collusion audit."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import torch
from torch import Tensor

from modelpact import __version__
from modelpact.adapters.base import GenerationPolicy as AdapterGenerationPolicy
from modelpact.adapters.tiny_lm import (
    TinyCausalLM,
    TinyConfig,
    TinyModelAdapter,
    save_tiny_checkpoint,
)
from modelpact.audit.active import AuditConfig, AuditResult, SubsetEvaluation, audit_patch_pool
from modelpact.audit.subsets import PatchSubset, enumerate_subsets
from modelpact.contracts import (
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
    canonical_contract_json,
)
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.patch.ast import DeltaProgram, LowRankMatrixDelta, Sum
from modelpact.patch.bundle import PatchBundle, create_patch_bundle, load_patch_bundle
from modelpact.patch.mount import mount_patch
from modelpact.status import VerificationOutcome
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical

_PATCH_IDS = tuple(f"behavior-{index}" for index in range(6))
_COLLUDERS = _PATCH_IDS[:3]
_TARGET_PROMPTS = tuple(f"synthetic behavior {index}:" for index in range(6))
_TRIGGER_PROMPT = "synthetic combined format:"
_CONTROL_PROMPT = "synthetic unchanged control:"
_ALL_PROMPTS = (*_TARGET_PROMPTS, _TRIGGER_PROMPT, _CONTROL_PROMPT)


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(canonical_dumps(dict(row)) + "\n" for row in rows).encode()


def _token_margin(logits: Tensor, expected: int) -> float:
    selected = float(logits[expected].item())
    competitors = torch.cat((logits[:expected], logits[expected + 1 :]))
    return selected - float(competitors.max().item())


def _evaluation_payload(evaluation: SubsetEvaluation) -> dict[str, object]:
    return {
        "subset": list(evaluation.subset),
        "outcome": evaluation.outcome.value,
        "margins": dict(sorted(evaluation.margins.items())),
        "violated_contracts": list(evaluation.violated_contracts),
        "metadata": dict(evaluation.metadata),
        "result_hash": evaluation.result_hash,
    }


def _audit_payload(result: AuditResult) -> dict[str, object]:
    return {
        "patch_ids": list(result.patch_ids),
        "possible_nonempty_subsets": result.possible_nonempty_subsets,
        "executed_subsets": result.executed_subset_count,
        "total_model_executions": result.total_model_executions,
        "claims": [item.value for item in result.claims],
        "coverage": result.coverage.value,
        "failing_subsets": [list(item) for item in result.failing_subsets],
        "minimal_failures": [list(item.reduced) for item in result.minimal_failures],
        "failure_reductions": [
            {
                "original": list(item.original),
                "reduced": list(item.reduced),
                "tested_candidates": [list(candidate) for candidate in item.tested_candidates],
                "one_minimal": item.one_minimal,
                "budget_exhausted": item.budget_exhausted,
            }
            for item in result.reduction_attempts
        ],
        "active_proposals": [
            {
                "subset": list(item.subset),
                "score": item.score,
                "predicted_worst_margin": item.predicted_worst_margin,
                "predictive_uncertainty": item.predictive_uncertainty,
                "novelty": item.novelty,
                "unexplored_interaction_fraction": item.unexplored_interaction_fraction,
            }
            for item in result.active_proposals
        ],
        "surrogate_fits": [
            {
                "contract_id": item.contract_id,
                "observations": item.observations,
                "selected_alpha": item.selected_alpha,
                "cross_validation_mse": item.cross_validation_mse,
                "degree": item.degree,
            }
            for item in result.surrogate_fits
        ],
        "search_space_exhausted": result.search_space_exhausted,
        "budget_exhausted": result.budget_exhausted,
    }


@dataclass(slots=True)
class _TinyPatchPool:
    root: Path
    adapter: TinyModelAdapter
    model: TinyCausalLM
    manifest: ModelManifest
    bundles: Mapping[str, PatchBundle]
    target_tokens: Mapping[str, int]
    trigger_expected_token: int
    trigger_failure_token: int
    control_expected_token: int
    compilation_residuals: Mapping[str, float]
    base_lm_head: Tensor
    oracle_calls: int = 0

    @classmethod
    def create(cls, root: Path) -> _TinyPatchPool:
        root.mkdir(parents=True, exist_ok=False)
        adapter = TinyModelAdapter()
        base = TinyCausalLM(
            TinyConfig(
                hidden_size=32,
                intermediate_size=48,
                num_layers=1,
                num_heads=4,
                max_sequence_length=64,
                tie_word_embeddings=False,
                initialization_seed=4301,
            )
        )
        checkpoint = save_tiny_checkpoint(base, root / "base-checkpoint")
        loaded = adapter.load(str(checkpoint), device="cpu", dtype=torch.float32)
        adapter.prepare(loaded)
        manifest = build_model_manifest(
            loaded,
            checkpoint=checkpoint,
            adapter_id=adapter.adapter_id,
        )
        batch = adapter.tokenizer().batch(_ALL_PROMPTS)
        with torch.inference_mode():
            output = loaded(
                batch.input_ids,
                batch.attention_mask,
                output_hidden_states=True,
            )
        positions = batch.attention_mask.to(torch.long).sum(dim=1) - 1
        rows = torch.arange(len(_ALL_PROMPTS))
        hidden = output.hidden_states[-1][rows, positions].detach().to(torch.float64)
        logits = output.logits[rows, positions].detach().to(torch.float64)
        if int(torch.linalg.matrix_rank(hidden).item()) < len(_ALL_PROMPTS):
            raise RuntimeError("Tiny composition prompt representations are rank deficient")

        base_tokens = tuple(int(row.argmax().item()) for row in logits)
        excluded = set(base_tokens)
        target_candidates = [adapter.tokenizer().byte_offset + value for value in b"ABCDEFUVWXYZ"]
        target_token_list: list[int] = []
        for candidate in target_candidates:
            if candidate not in excluded and candidate not in target_token_list:
                target_token_list.append(candidate)
            if len(target_token_list) == len(_PATCH_IDS):
                break
        if len(target_token_list) != len(_PATCH_IDS):
            raise RuntimeError("could not select distinct tiny benchmark target tokens")

        trigger_logits = logits[-2]
        trigger_expected = int(trigger_logits.argmax().item())
        ordered = torch.argsort(trigger_logits, descending=True).tolist()
        trigger_failure = next(
            int(token) for token in ordered[1:] if int(token) not in set(target_token_list)
        )
        trigger_margin = float(
            (trigger_logits[trigger_expected] - trigger_logits[trigger_failure]).item()
        )
        if trigger_margin <= 0:
            raise RuntimeError("Tiny composition trigger has no strict base margin")
        collusion_increment = 0.4 * trigger_margin

        constraints = hidden
        trigger_values = torch.zeros(len(_ALL_PROMPTS), dtype=torch.float64)
        trigger_values[-2] = 1.0
        trigger_direction = torch.linalg.lstsq(constraints, trigger_values).solution
        bundles: dict[str, PatchBundle] = {}
        target_tokens: dict[str, int] = {}
        residuals: dict[str, float] = {}
        requirements = ModelRequirements(
            tokenizer_hash=manifest.signature.tokenizer_hash,
            base_signature=manifest.signature.signature_hash,
            architecture_hash=manifest.signature.architecture_hash,
            state_schema_hash=manifest.signature.state_schema_hash,
            adapter_id=manifest.signature.adapter_id,
        )
        for index, patch_id in enumerate(_PATCH_IDS):
            desired = torch.zeros(len(_ALL_PROMPTS), dtype=torch.float64)
            desired[index] = 1.0
            target_direction = torch.linalg.lstsq(constraints, desired).solution
            directions = [target_direction]
            if patch_id in _COLLUDERS:
                directions.append(trigger_direction)
            right = torch.stack(directions).to(torch.float32)
            left = torch.zeros((loaded.config.vocab_size, len(directions)), dtype=torch.float32)
            target_token = target_token_list[index]
            target_tokens[patch_id] = target_token
            target_boost = (
                max(
                    0.0,
                    float(logits[index].max().item() - logits[index, target_token].item()),
                )
                + 0.5
            )
            left[target_token, 0] = target_boost
            if patch_id in _COLLUDERS:
                left[trigger_failure, 1] = collusion_increment
            residuals[patch_id] = float(
                (
                    constraints @ right.to(torch.float64).mT
                    - torch.stack(
                        (desired, trigger_values) if patch_id in _COLLUDERS else (desired,)
                    ).mT
                )
                .abs()
                .max()
                .item()
            )
            contract = BehaviorContract(
                schema_version=1,
                id=f"tiny-composition-{patch_id}",
                contract_version=1,
                model_requirements=requirements,
                objectives=(
                    CompileObjective(
                        f"{patch_id}-activation-fit",
                        ObjectiveType.ACTIVATION_DIRECTION,
                        "probes/target.jsonl",
                    ),
                ),
                targets=(
                    VerificationAssertion(
                        f"{patch_id}-generation",
                        AssertionType.FREE_GENERATION_MATCH,
                        "probes/target.jsonl",
                        {"minimum_pass_rate": 1.0},
                    ),
                ),
                guards=(
                    VerificationAssertion(
                        f"{patch_id}-guards",
                        AssertionType.FREE_GENERATION_MATCH,
                        "probes/guards.jsonl",
                        {"minimum_pass_rate": 1.0},
                    ),
                ),
                holdout=HoldoutPolicy(),
                statistics=StatisticsPolicy(bootstrap_samples=64, bootstrap_seed=9300 + index),
                generation=GenerationPolicy(
                    mode=GenerationMode.GREEDY,
                    max_new_tokens=1,
                    seeds=(0,),
                ),
                description="Finite TinyCausalLM composition benchmark contract.",
            )
            target_text = adapter.tokenizer().decode((target_token,))
            guard_rows = (
                {
                    "id": "collusion-trigger",
                    "prompt": _TRIGGER_PROMPT,
                    "expected": adapter.tokenizer().decode((trigger_expected,)),
                },
                {
                    "id": "unchanged-control",
                    "prompt": _CONTROL_PROMPT,
                    "expected": adapter.tokenizer().decode((base_tokens[-1],)),
                },
            )
            bundle = create_patch_bundle(
                root / "patches" / patch_id,
                name=patch_id,
                base_signature=manifest.signature.to_dict(),
                state_schema=manifest.state_schema,
                program=DeltaProgram({"lm_head.weight": LowRankMatrixDelta("left", "right")}),
                tensors={"left": left, "right": right},
                tool_version=__version__,
                contracts={
                    "contracts/target.json": (canonical_contract_json(contract) + "\n").encode(),
                    "contracts/probes/target.jsonl": _jsonl(
                        (
                            {
                                "id": f"{patch_id}-target",
                                "prompt": _TARGET_PROMPTS[index],
                                "expected": target_text,
                            },
                        )
                    ),
                    "contracts/probes/guards.jsonl": _jsonl(guard_rows),
                },
                provides=(contract.contract_id,),
                preserves=(contract.contract_id,),
                compiler_configuration={
                    "algorithm": "deterministic_contract_direction_lstsq",
                    "rank": len(directions),
                    "target_prompt_count": 1,
                    "guard_prompt_count": len(guard_rows),
                    "residual": residuals[patch_id],
                },
            )
            bundles[patch_id] = load_patch_bundle(bundle.path)
        if max(residuals.values()) > 1e-4:
            raise RuntimeError("Tiny composition low-rank compilation residual is too large")
        return cls(
            root=root,
            adapter=adapter,
            model=loaded,
            manifest=manifest,
            bundles=bundles,
            target_tokens=target_tokens,
            trigger_expected_token=trigger_expected,
            trigger_failure_token=trigger_failure,
            control_expected_token=base_tokens[-1],
            compilation_residuals=residuals,
            base_lm_head=loaded.lm_head.weight.detach().clone(),
        )

    def _combined(self, subset: PatchSubset) -> tuple[DeltaProgram, dict[str, Tensor]]:
        terms = []
        tensors: dict[str, Tensor] = {}
        for index, patch_id in enumerate(subset):
            bundle = self.bundles[patch_id]
            operation = bundle.program.targets["lm_head.weight"]
            if not isinstance(operation, LowRankMatrixDelta):
                raise RuntimeError("Tiny composition bundle is not low rank")
            left_name = f"p{index}_left"
            right_name = f"p{index}_right"
            tensors[left_name] = bundle.tensors[operation.left]
            tensors[right_name] = bundle.tensors[operation.right]
            terms.append(LowRankMatrixDelta(left_name, right_name, operation.scale))
        if not terms:
            raise ValueError("empty subset has no combined delta")
        expression = terms[0] if len(terms) == 1 else Sum(tuple(terms))
        return DeltaProgram({"lm_head.weight": expression}), tensors

    def evaluate(self, subset: PatchSubset) -> SubsetEvaluation:
        self.oracle_calls += 1
        session = None
        if subset:
            program, tensors = self._combined(subset)
            session = mount_patch(
                self.model,
                program,
                tensors,
                state_schema=self.manifest.state_schema,
            )
        try:
            batch = self.adapter.tokenizer().batch(_ALL_PROMPTS)
            with torch.inference_mode():
                logits = self.adapter.forward_logits(self.model, batch)
            positions = batch.attention_mask.to(torch.long).sum(dim=1) - 1
            rows = torch.arange(len(_ALL_PROMPTS))
            last_logits = logits[rows, positions]
            samples = self.adapter.generate(
                self.model,
                batch,
                AdapterGenerationPolicy(mode="greedy", max_new_tokens=1, seed=0),
            )
        finally:
            if session is not None:
                session.unmount()
        if not torch.equal(self.model.lm_head.weight.detach(), self.base_lm_head):
            raise RuntimeError("Tiny composition runtime unmount changed the base model")

        margins: dict[str, float] = {
            "guard:collusion-trigger": _token_margin(last_logits[-2], self.trigger_expected_token),
            "guard:unchanged-control": _token_margin(last_logits[-1], self.control_expected_token),
        }
        for patch_id in subset:
            index = _PATCH_IDS.index(patch_id)
            margins[f"target:{patch_id}"] = _token_margin(
                last_logits[index], self.target_tokens[patch_id]
            )
        generated = {
            prompt: (sample.token_ids[0] if sample.token_ids else None)
            for prompt, sample in zip(_ALL_PROMPTS, samples, strict=True)
        }
        violated_set = {key for key, value in margins.items() if value < 0.0}
        if generated[_TRIGGER_PROMPT] != self.trigger_expected_token:
            violated_set.add("guard:collusion-trigger")
        if generated[_CONTROL_PROMPT] != self.control_expected_token:
            violated_set.add("guard:unchanged-control")
        for patch_id in subset:
            index = _PATCH_IDS.index(patch_id)
            if generated[_TARGET_PROMPTS[index]] != self.target_tokens[patch_id]:
                violated_set.add(f"target:{patch_id}")
        violated = tuple(sorted(violated_set))
        metadata: dict[str, object] = {
            "adapter_id": self.adapter.adapter_id,
            "base_signature": self.manifest.signature.signature_hash,
            "generated_token_ids": generated,
            "runtime_mount": "PYTORCH_PARAMETRIZATION",
            "runtime_unmount_exact": True,
            "trigger_expected_token": self.trigger_expected_token,
            "trigger_failure_token": self.trigger_failure_token,
        }
        payload = {
            "subset": list(subset),
            "margins": dict(sorted(margins.items())),
            "violated_contracts": list(violated),
            "generated_token_ids": generated,
        }
        return SubsetEvaluation(
            subset,
            margins,
            VerificationOutcome.PASS if not violated else VerificationOutcome.FAIL,
            violated,
            metadata,
            hash_canonical(payload),
        )

    def patch_rows(self) -> list[dict[str, object]]:
        rows = []
        for patch_id in _PATCH_IDS:
            bundle = self.bundles[patch_id]
            operation = cast(LowRankMatrixDelta, bundle.program.targets["lm_head.weight"])
            rows.append(
                {
                    "patch_id": patch_id,
                    "content_id": bundle.manifest.patch_id,
                    "contract_hashes": list(bundle.manifest.provides),
                    "rank": int(bundle.tensors[operation.right].shape[0]),
                    "active_modules": ["lm_head.weight"],
                    "factor_parameters": int(
                        bundle.tensors[operation.left].numel()
                        + bundle.tensors[operation.right].numel()
                    ),
                    "compilation_residual": self.compilation_residuals[patch_id],
                }
            )
        return rows


ResultT = TypeVar("ResultT", bound=dict[str, object])


def _with_fixture(
    output: str | Path | None,
    operation: Callable[[_TinyPatchPool], ResultT],
) -> ResultT:
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if output is None:
            with tempfile.TemporaryDirectory(prefix="modelpact-tiny-composition-") as temporary:
                return operation(_TinyPatchPool.create(Path(temporary) / "fixture"))
        root = Path(output)
        if root.exists():
            raise FileExistsError(root)
        fixture = _TinyPatchPool.create(root)
        result = operation(fixture)
        atomic_write_text(root / "result.json", canonical_dumps(result) + "\n")
        return result
    finally:
        torch.set_num_threads(prior_threads)


def run_closure_matrix(output: str | Path | None = None) -> dict[str, object]:
    """Execute all 63 nonempty stacks on real TinyCausalLM patch bundles."""

    def execute(pool: _TinyPatchPool) -> dict[str, object]:
        started = time.perf_counter()
        audit = audit_patch_pool(
            _PATCH_IDS,
            oracle=pool.evaluate,
            config=AuditConfig(
                subset_budget=63,
                exhaustive_threshold=6,
                bootstrap_samples=4,
                seed=4401,
            ),
        )
        evaluations = {item.subset: item for item in audit.evaluations}
        singletons_pass = all(evaluations[(patch_id,)].passed for patch_id in _PATCH_IDS)
        expected_failures = {
            subset for subset in enumerate_subsets(_PATCH_IDS) if set(_COLLUDERS) <= set(subset)
        }
        observed_failures = {item.subset for item in audit.evaluations if not item.passed}
        success = bool(
            singletons_pass
            and audit.search_space_exhausted
            and audit.executed_subset_count == 63
            and observed_failures == expected_failures
        )
        return {
            "schema_version": 1,
            "suite": "ModelPactBench",
            "benchmark": "Contract Closure Matrix",
            "model": "TinyCausalLM",
            "execution": "real_runtime_mounted_behavior_patch_bundles",
            "status": "PASS" if success else "FAIL",
            "success": success,
            "individual_patches_pass": singletons_pass,
            "patches": pool.patch_rows(),
            "ground_truth_failure_count": len(observed_failures),
            "minimal_interaction_order": min(map(len, observed_failures), default=None),
            "subset_ground_truth": [
                _evaluation_payload(item)
                for item in sorted(audit.evaluations, key=lambda row: (len(row.subset), row.subset))
            ],
            "performance": {
                "wall_seconds": time.perf_counter() - started,
                "oracle_calls": pool.oracle_calls,
                "forward_executions": pool.oracle_calls,
                "generation_executions": pool.oracle_calls,
            },
            **_audit_payload(audit),
        }

    return _with_fixture(output, execute)


def _baseline(
    pool: _TinyPatchPool,
    candidates: Sequence[PatchSubset],
    ground_truth_failures: set[PatchSubset],
    *,
    budget: int,
) -> dict[str, object]:
    started = time.perf_counter()
    evaluations = [pool.evaluate(subset) for subset in tuple(candidates)[:budget]]
    failures = [item.subset for item in evaluations if not item.passed]
    first_failure_index = next(
        (index for index, item in enumerate(evaluations, start=1) if not item.passed), None
    )
    discovered = set(failures) & ground_truth_failures
    return {
        "executions": len(evaluations),
        "failure_found": bool(failures),
        "first_failure": None if not failures else list(failures[0]),
        "time_to_first_failure_executions": first_failure_index,
        "failure_discovery_recall": (
            len(discovered) / len(ground_truth_failures) if ground_truth_failures else 1.0
        ),
        "false_assurance_rate": float(bool(ground_truth_failures) and not failures),
        "wall_seconds": time.perf_counter() - started,
    }


def run_benign_collusion(
    output: str | Path | None = None,
    *,
    subset_budget: int = 40,
) -> dict[str, object]:
    """Find and reduce a harmless three-way-only TinyCausalLM failure."""

    def execute(pool: _TinyPatchPool) -> dict[str, object]:
        started = time.perf_counter()
        all_subsets = enumerate_subsets(_PATCH_IDS)
        exact = tuple(pool.evaluate(subset) for subset in all_subsets)
        ground_truth_failures = {item.subset for item in exact if not item.passed}
        exact_by_subset = {item.subset: item for item in exact}
        singletons_pass = all(exact_by_subset[(patch_id,)].passed for patch_id in _PATCH_IDS)
        pairs = enumerate_subsets(_PATCH_IDS, minimum_order=2, maximum_order=2)
        pairs_pass = all(exact_by_subset[item].passed for item in pairs)

        active = audit_patch_pool(
            _PATCH_IDS,
            oracle=pool.evaluate,
            config=AuditConfig(
                subset_budget=subset_budget,
                exhaustive_threshold=0,
                surrogate_degree=3,
                include_all_pairs=True,
                initial_random_subsets=0,
                active_batch_size=1,
                bootstrap_samples=8,
                seed=4417,
            ),
        )
        active_minimal = {item.reduced for item in active.minimal_failures}
        random_generator = torch.Generator(device="cpu").manual_seed(4419)
        random_order = tuple(
            all_subsets[index]
            for index in torch.randperm(len(all_subsets), generator=random_generator).tolist()
        )
        overlap_order = tuple(
            sorted(
                all_subsets,
                key=lambda subset: (-len(subset) * (len(subset) - 1) // 2, subset),
            )
        )
        baselines = {
            "singleton_only": _baseline(
                pool,
                enumerate_subsets(_PATCH_IDS, maximum_order=1),
                ground_truth_failures,
                budget=subset_budget,
            ),
            "pairwise_only": _baseline(
                pool,
                pairs,
                ground_truth_failures,
                budget=subset_budget,
            ),
            "random_subsets": _baseline(
                pool, random_order, ground_truth_failures, budget=subset_budget
            ),
            "parameter_overlap": _baseline(
                pool, overlap_order, ground_truth_failures, budget=subset_budget
            ),
            "active_sparse_interaction": {
                "executions": active.executed_subset_count,
                "failure_found": bool(active.failing_subsets),
                "first_failure": (
                    None if not active.failing_subsets else list(active.failing_subsets[0])
                ),
                "failure_discovery_recall": len(set(active.failing_subsets) & ground_truth_failures)
                / len(ground_truth_failures),
                "false_assurance_rate": float(not active.failing_subsets),
                "time_to_first_failure_executions": next(
                    (
                        index
                        for index, item in enumerate(active.evaluations, start=1)
                        if not item.passed
                    ),
                    None,
                ),
                "minimal_failing_subset_recovered": _COLLUDERS in active_minimal,
            },
        }
        success = bool(
            singletons_pass
            and pairs_pass
            and _COLLUDERS in ground_truth_failures
            and active.failing_subsets
            and _COLLUDERS in active_minimal
            and active.executed_subset_count < len(all_subsets)
        )
        return {
            "schema_version": 1,
            "suite": "ModelPactBench",
            "benchmark": "Benign Collusion",
            "model": "TinyCausalLM",
            "execution": "real_runtime_mounted_behavior_patch_bundles",
            "status": "PASS" if success else "FAIL",
            "success": success,
            "individual_patches_pass": singletons_pass,
            "relevant_pairs_pass": pairs_pass,
            "three_way_ground_truth_fails": _COLLUDERS in ground_truth_failures,
            "ground_truth_failure_count": len(ground_truth_failures),
            "ground_truth_failing_subsets": [
                list(item) for item in sorted(ground_truth_failures, key=lambda x: (len(x), x))
            ],
            "subset_ground_truth": [
                _evaluation_payload(item)
                for item in sorted(exact, key=lambda row: (len(row.subset), row.subset))
            ],
            "expected_minimal_failure": list(_COLLUDERS),
            "patches": pool.patch_rows(),
            "baseline_comparison": baselines,
            "negative_findings": [
                (
                    "The parameter-overlap ordering found a failing superset sooner and "
                    "covered more failures than active search at the same budget; only "
                    "executed ddmin established the reported one-minimal triple."
                ),
                (
                    "The deterministic random baseline also found a failure in this seed; "
                    "this single fixture does not establish an active-search advantage."
                ),
            ],
            "overlap_evidence": {
                "all_patches_target_same_parameter": True,
                "pairwise_module_overlap_jaccard": 1.0,
                "ranking_rule": (
                    "descending count of equally overlapping patch pairs, then canonical subset"
                ),
                "interpretation": (
                    "Parameter overlap is a selection baseline, never a compatibility verdict."
                ),
            },
            "performance": {
                "wall_seconds": time.perf_counter() - started,
                "oracle_calls": pool.oracle_calls,
                "possible_subsets": len(all_subsets),
                "active_subset_budget": subset_budget,
            },
            **_audit_payload(active),
        }

    return _with_fixture(output, execute)


__all__ = ["run_benign_collusion", "run_closure_matrix"]
