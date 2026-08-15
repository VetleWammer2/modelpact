"""Fast executed ModelPactBench organisms for CPU CI.

These are deliberately transparent neural organisms, not mocked command output.
Every subset margin comes from a PyTorch forward pass and every repair/rebase
candidate comes from a new optimization. Larger causal-LM ModelPactBench workflows
live beside these exact-ground-truth experiments.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from modelpact.audit.active import (
    AuditConfig,
    AuditResult,
    SubsetEvaluation,
    audit_patch_pool,
)
from modelpact.audit.subsets import PatchSubset, enumerate_subsets
from modelpact.baselines.merging import (
    cat_projection,
    dare,
    task_arithmetic,
    ties_merge,
    weighted_delta_sum,
)
from modelpact.compose.closure import (
    ContractMargin,
    MarginKind,
    PatchOperand,
    VerificationReport,
)
from modelpact.compose.merge import (
    JointCompilationResult,
    MergeBudget,
    SemanticMergeRequest,
    semantic_merge,
)
from modelpact.rebase.compile import (
    BehavioralRecompileRequest,
    BehavioralRecompileResult,
    RebaseBudget,
    RebaseRequest,
    TeacherContext,
    semantic_rebase,
)
from modelpact.rebase.direct import (
    BaseModelDescriptor,
    RebasePatch,
    RebaseVerification,
)
from modelpact.status import VerificationOutcome


class ScalarBehaviorModel(nn.Module):
    def __init__(self, base_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[base_weight]], dtype=torch.float64))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.to(torch.float64) @ self.weight.mT


def _scalar_execution(delta: float) -> float:
    model = ScalarBehaviorModel()
    with torch.no_grad():
        model.weight.add_(delta)
    return float(model(torch.ones((1, 1), dtype=torch.float64)).item())


def _audit_dict(result: AuditResult) -> dict[str, object]:
    # AuditResult contains enums and mappings; construct a stable research row
    # rather than relying on repr or JSON fallback coercion.
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
        "search_space_exhausted": result.search_space_exhausted,
        "budget_exhausted": result.budget_exhausted,
    }


def run_scalar_closure_smoke() -> dict[str, object]:
    """Execute a non-mandatory scalar closure algorithm smoke test."""

    patch_ids = tuple(f"patch-{index}" for index in range(6))
    deltas = {patch_id: 0.11 + 0.025 * index for index, patch_id in enumerate(patch_ids)}
    executions = 0

    def oracle(subset: PatchSubset) -> SubsetEvaluation:
        nonlocal executions
        executions += 1
        output = _scalar_execution(sum(deltas[item] for item in subset))
        margins: dict[str, float] = {}
        for patch_id in subset:
            margins[f"{patch_id}:target"] = output - 0.05
            margins[f"{patch_id}:guard"] = 0.48 - output
        if not subset:
            margins["base:guard"] = 0.48 - output
        violated = tuple(sorted(key for key, margin in margins.items() if margin < 0))
        return SubsetEvaluation(
            subset,
            margins,
            VerificationOutcome.PASS if not violated else VerificationOutcome.FAIL,
            violated,
            {"model_output": output},
        )

    started = time.perf_counter()
    result = audit_patch_pool(
        patch_ids,
        oracle=oracle,
        config=AuditConfig(subset_budget=63, exhaustive_threshold=6, seed=17),
    )
    individual_pass = all(oracle((patch_id,)).passed for patch_id in patch_ids)
    success = (
        individual_pass and result.search_space_exhausted and result.executed_subset_count == 63
    )
    return {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": "Scalar closure algorithm smoke",
        "model": "executed_scalar_pytorch_organism",
        "individual_patches_pass": individual_pass,
        "status": "PASS" if success else "FAIL",
        "success": success,
        "oracle_executions_including_postcheck": executions,
        "wall_seconds": time.perf_counter() - started,
        **_audit_dict(result),
    }


def run_scalar_collusion_smoke(*, subset_budget: int = 13) -> dict[str, object]:
    """Execute a non-mandatory scalar active-audit smoke test."""

    patch_ids = ("field-a", "field-b", "field-c", "unrelated-d")
    deltas = {"field-a": 0.34, "field-b": 0.34, "field-c": 0.34, "unrelated-d": 0.05}
    static_high_risk = tuple(sorted(patch_ids, key=lambda item: (-abs(deltas[item]), item))[:3])

    def oracle(subset: PatchSubset) -> SubsetEvaluation:
        output = _scalar_execution(sum(deltas[item] for item in subset))
        margins = {f"{item}:target": output - 0.02 for item in subset}
        margins["required-output-field"] = 1.0 - output
        violated = tuple(sorted(key for key, margin in margins.items() if margin < 0))
        return SubsetEvaluation(
            subset,
            margins,
            VerificationOutcome.PASS if not violated else VerificationOutcome.FAIL,
            violated,
            {"model_output": output, "failure_semantics": "synthetic required field suppression"},
        )

    relevant_pairs_pass = all(
        oracle(subset).passed
        for subset in enumerate_subsets(patch_ids, minimum_order=2, maximum_order=2)
    )
    all_subsets = enumerate_subsets(patch_ids)

    def baseline_summary(candidates: tuple[PatchSubset, ...]) -> dict[str, object]:
        executed: list[PatchSubset] = []
        failure: PatchSubset | None = None
        for subset in candidates[:subset_budget]:
            executed.append(subset)
            if not oracle(subset).passed:
                failure = subset
                break
        return {
            "executions": len(executed),
            "failure_found": failure is not None,
            "first_failure": None if failure is None else list(failure),
        }

    singleton_pair_design = enumerate_subsets(patch_ids, maximum_order=2)
    generator = torch.Generator(device="cpu").manual_seed(29)
    permutation = torch.randperm(len(all_subsets), generator=generator).tolist()
    random_design = tuple(all_subsets[index] for index in permutation)
    parameter_overlap_design = tuple(sorted(all_subsets, key=lambda item: (-len(item), item)))
    started = time.perf_counter()
    result = audit_patch_pool(
        patch_ids,
        oracle=oracle,
        config=AuditConfig(
            subset_budget=subset_budget,
            exhaustive_threshold=0,
            surrogate_degree=3,
            include_all_pairs=True,
            initial_random_subsets=1,
            bootstrap_samples=4,
            seed=29,
        ),
        high_risk=(static_high_risk,),
    )
    individual_pass = all(oracle((item,)).passed for item in patch_ids)
    three_way_fails = not oracle(("field-a", "field-b", "field-c")).passed
    success = (
        individual_pass
        and relevant_pairs_pass
        and three_way_fails
        and bool(result.failing_subsets)
        and bool(result.minimal_failures)
    )
    return {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": "Scalar collusion algorithm smoke",
        "model": "executed_scalar_pytorch_organism",
        "individual_patches_pass": individual_pass,
        "relevant_pairs_pass": relevant_pairs_pass,
        "static_high_risk_subset": list(static_high_risk),
        "three_way_ground_truth_fails": three_way_fails,
        "status": "PASS" if success else "FAIL",
        "success": success,
        "baseline_comparison": {
            "singleton_pair_only": baseline_summary(singleton_pair_design),
            "random": baseline_summary(random_design),
            "parameter_overlap": baseline_summary(parameter_overlap_design),
            "active_sparse_interaction": {
                "executions": result.executed_subset_count,
                "failure_found": bool(result.failing_subsets),
                "first_failure": (
                    None if not result.failing_subsets else list(result.failing_subsets[0])
                ),
            },
        },
        "wall_seconds": time.perf_counter() - started,
        **_audit_dict(result),
    }


@dataclass(frozen=True, slots=True)
class _MergeExecutor:
    minimum_target: float = 0.5
    maximum_guard: float = 1.0

    def __call__(
        self,
        delta: Mapping[str, Tensor],
        contract_ids: tuple[str, ...],
    ) -> VerificationReport:
        output = _scalar_execution(float(delta["weight"].item()))
        margins: list[ContractMargin] = []
        for contract_id in contract_ids:
            margins.append(
                ContractMargin(contract_id, MarginKind.TARGET, output - self.minimum_target)
            )
            margins.append(
                ContractMargin(
                    f"{contract_id}:guard",
                    MarginKind.GUARD,
                    self.maximum_guard - output,
                )
            )
        outcome = (
            VerificationOutcome.PASS
            if all(item.passed for item in margins)
            else VerificationOutcome.FAIL
        )
        return VerificationReport(outcome, tuple(margins))


def _joint_scalar_compiler(request: SemanticMergeRequest) -> JointCompilationResult:
    parameter = nn.Parameter(request.initial_delta["weight"].detach().to(torch.float64).clone())
    optimizer = torch.optim.Adam([parameter], lr=0.08)
    best: Tensor | None = None
    best_margin = -float("inf")
    steps = 0
    for step in range(1, request.budget.maximum_steps + 1):
        steps = step
        optimizer.zero_grad(set_to_none=True)
        output = parameter.reshape(())
        target_violation = torch.relu(torch.tensor(0.5, dtype=output.dtype) - output)
        guard_violation = torch.relu(output - torch.tensor(1.0, dtype=output.dtype))
        initial = request.initial_delta["weight"].reshape(()).to(output)
        proximity = 0.01 * (output - initial).square()
        loss = target_violation.square() + guard_violation.square() + proximity
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub gap.
        optimizer.step()
        margin = min(float(output.detach().item() - 0.5), float(1.0 - output.detach().item()))
        if margin >= 0 and margin > best_margin:
            best = parameter.detach().clone()
            best_margin = margin
    return JointCompilationResult(
        candidate_delta=None if best is None else {"weight": best},
        optimization_succeeded=best is not None,
        budget_exhausted=best is None,
        steps_executed=steps,
        restarts_executed=1,
        best_margins=dict.fromkeys(request.contract_ids, best_margin),
        violated_contracts=() if best is not None else request.contract_ids,
        diagnostics={"new_optimization": True, "optimizer": "Adam"},
    )


def _joint_multitask_scalar_baseline(*, steps: int) -> tuple[dict[str, Tensor], int]:
    """Train an unconstrained scalar baseline under the same step budget."""

    parameter = nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.08)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = parameter.reshape(())
        target_loss = (output - torch.tensor(0.7, dtype=output.dtype)).square()
        target_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub gap.
        optimizer.step()
    return {"weight": parameter.detach().clone()}, steps


def run_semantic_merge() -> dict[str, object]:
    operands = (
        PatchOperand(
            "parent-a",
            "base",
            "schema",
            {"weight": torch.tensor([[0.7]], dtype=torch.float64)},
            ("contract-a",),
        ),
        PatchOperand(
            "parent-b",
            "base",
            "schema",
            {"weight": torch.tensor([[0.7]], dtype=torch.float64)},
            ("contract-b",),
        ),
    )
    started = time.perf_counter()
    result = semantic_merge(
        operands,
        executor=_MergeExecutor(),
        compiler=_joint_scalar_compiler,
        budget=MergeBudget(maximum_steps=120),
    )
    states = [dict(operand.delta) for operand in operands]
    joint_baseline, joint_steps = _joint_multitask_scalar_baseline(steps=120)
    baseline_deltas = {
        "naive_delta_sum": weighted_delta_sum(states),
        "weighted_delta_sum": weighted_delta_sum(states, (0.5, 0.5)),
        "task_arithmetic": task_arithmetic(states),
        "ties": ties_merge(states, density=1.0),
        "dare": dare(states, drop_probability=0.5, seed=31),
        "cat_style_projection": cat_projection(states),
        "joint_multitask_low_rank": joint_baseline,
    }
    executor = _MergeExecutor()
    baseline_comparison: dict[str, dict[str, object]] = {}
    for name, delta in sorted(baseline_deltas.items()):
        report = executor(delta, ("contract-a", "contract-b"))
        baseline_comparison[name] = {
            "delta": float(delta["weight"].item()),
            "outcome": report.outcome.value,
            "passed": report.outcome is VerificationOutcome.PASS,
            "steps": joint_steps if name == "joint_multitask_low_rank" else 0,
        }
    passing_baselines = sorted(
        name for name, row in baseline_comparison.items() if row["passed"] is True
    )
    parents_pass = all(
        _MergeExecutor()(dict(operand.delta), operand.contract_ids).outcome
        is VerificationOutcome.PASS
        for operand in operands
    )
    success = (
        parents_pass
        and not result.naive_composition.closed
        and result.compiler_invoked
        and result.verified
        and result.compilation is not None
        and result.compilation.steps_executed > 0
    )
    return {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": "Semantic Merge",
        "model": "executed_scalar_pytorch_organism",
        "parents_individually_pass": parents_pass,
        "naive_claim": result.naive_composition.claim.value,
        "naive_passed": result.naive_composition.closed,
        "compiler_invoked": result.compiler_invoked,
        "disposition": result.disposition.value,
        "claim": result.claim.value,
        "merged_verified": result.verified,
        "status": "PASS" if success else "FAIL",
        "success": success,
        "merged_delta": float(result.delta["weight"].item()) if result.delta else None,
        "optimization_steps": result.compilation.steps_executed if result.compilation else 0,
        "baseline_comparison": baseline_comparison,
        "negative_findings": (
            [
                "At least one parameter-space or joint-training baseline also passed this "
                f"finite union contract: {', '.join(passing_baselines)}."
            ]
            if passing_baselines
            else []
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def run_semantic_rebase(*, cross_architecture: bool = False) -> dict[str, object]:
    source_base = BaseModelDescriptor(
        "base-v1",
        "arch-linear",
        "schema-v1",
        "tok",
        "causal_lm",
        {"weight": (1, 2)},
        "tiny-family",
    )
    target_base = BaseModelDescriptor(
        "base-v2",
        "arch-mlp" if cross_architecture else "arch-linear",
        "schema-v2" if cross_architecture else "schema-v1",
        "tok",
        "causal_lm",
        {"weight": (1, 2)},
        "tiny-family",
    )
    patch = RebasePatch(
        "behavior-patch-v1",
        "base-v1",
        {"weight": torch.tensor([[0.7, 0.0]], dtype=torch.float64)},
        ("target",),
        ("old-guard",),
    )
    new_base_weight = torch.tensor([[0.4, 0.8]], dtype=torch.float64)

    def apply_direct(delta: Mapping[str, Tensor], target: BaseModelDescriptor) -> object:
        del target
        return new_base_weight + delta["weight"]

    def verify(
        candidate: object,
        target_contract_ids: tuple[str, ...],
        guard_contract_ids: tuple[str, ...],
    ) -> RebaseVerification:
        weight = cast(Tensor, candidate)
        target_output = float(weight[0, 0].item())
        control_output = float(weight[0, 1].item())
        target_margins = {
            identifier: 0.05 - abs(target_output - 0.7) for identifier in target_contract_ids
        }
        guard_margins = {
            identifier: 0.05 - abs(control_output - 0.8) for identifier in guard_contract_ids
        }
        all_margins = (*target_margins.values(), *guard_margins.values())
        return RebaseVerification(
            (
                VerificationOutcome.PASS
                if all(value >= 0 for value in all_margins)
                else VerificationOutcome.FAIL
            ),
            target_margins,
            guard_margins,
        )

    def teacher_builder(request: RebaseRequest) -> TeacherContext:
        del request
        return TeacherContext(
            old_patched_teacher=torch.tensor([[0.7, 0.0]], dtype=torch.float64),
            new_unpatched_teacher=new_base_weight.clone(),
            old_behavior_margins={"target": 0.05},
            evidence_count=2,
        )

    def recompile(request: BehavioralRecompileRequest) -> BehavioralRecompileResult:
        delta = nn.Parameter(torch.zeros_like(new_base_weight))
        optimizer = torch.optim.Adam([delta], lr=0.08)
        best: Tensor | None = None
        steps = 0
        for step in range(1, request.budget.maximum_steps + 1):
            steps = step
            optimizer.zero_grad(set_to_none=True)
            patched = new_base_weight + delta
            target_loss = (patched[0, 0] - 0.7).square()
            guard_loss = (patched[0, 1] - new_base_weight[0, 1]).square()
            complexity = 1e-3 * delta.square().sum()
            combined_loss = target_loss + 10.0 * guard_loss + complexity
            combined_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub gap.
            optimizer.step()
            if (
                abs(float(patched[0, 0].detach().item()) - 0.7) <= 0.05
                and abs(float(patched[0, 1].detach().item()) - 0.8) <= 0.05
            ):
                best = delta.detach().clone()
        target_margin = (
            -1.0 if best is None else 0.05 - abs(float((new_base_weight + best)[0, 0]) - 0.7)
        )
        guard_margin = (
            -1.0 if best is None else 0.05 - abs(float((new_base_weight + best)[0, 1]) - 0.8)
        )
        return BehavioralRecompileResult(
            None if best is None else {"weight": best},
            best is not None,
            best is None,
            steps,
            1,
            {"target": target_margin},
            {"new-base-control": guard_margin},
            () if best is not None else ("target",),
            {} if best is None else {"parameters": int(best.numel()), "norm": float(best.norm())},
        )

    def rebase_verifier(
        candidate: object,
        target_contract_ids: tuple[str, ...],
        guard_contract_ids: tuple[str, ...],
    ) -> RebaseVerification:
        if isinstance(candidate, dict):
            weight = new_base_weight + cast(dict[str, Tensor], candidate)["weight"]
        else:
            weight = cast(Tensor, candidate)
        return verify(weight, target_contract_ids, guard_contract_ids)

    request = RebaseRequest(
        patch,
        source_base,
        target_base,
        ("new-base-control",),
        RebaseBudget(maximum_steps=100),
        allow_cross_architecture=True,
    )
    started = time.perf_counter()
    result = semantic_rebase(
        request,
        applier=apply_direct,
        verifier=rebase_verifier,
        teacher_builder=teacher_builder,
        recompiler=recompile,
    )
    success = (
        result.direct_transfer.attempted != cross_architecture
        and not result.direct_transfer.verified
        and result.recompile is not None
        and result.recompile.steps_executed > 0
        and result.verified
    )
    return {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": (
            "RebaseBench cross-architecture" if cross_architecture else "RebaseBench same-family"
        ),
        "model": "executed_two_feature_pytorch_organism",
        "direct_attempted": result.direct_transfer.attempted,
        "direct_verified": result.direct_transfer.verified,
        "recompiled": result.recompile is not None,
        "claim": result.claim.value,
        "disposition": result.disposition.value,
        "verified": result.verified,
        "status": "PASS" if success else "FAIL",
        "success": success,
        "optimization_steps": result.recompile.steps_executed if result.recompile else 0,
        "target_margins": dict(result.verification.target_margins) if result.verification else {},
        "guard_margins": dict(result.verification.guard_margins) if result.verification else {},
        "wall_seconds": time.perf_counter() - started,
    }


def _fit_polynomial(
    examples: tuple[tuple[float, float], ...],
    *,
    steps: int = 250,
) -> tuple[Tensor, tuple[float, ...]]:
    weights = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([weights], lr=0.05)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.tensor([item[0] for item in examples], dtype=torch.float64)
        targets = torch.tensor([item[1] for item in examples], dtype=torch.float64)
        predictions = weights[0] * inputs + weights[1] * inputs.square()
        loss = (predictions - targets).square().mean() + 1e-4 * weights.square().sum()
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub gap.
        optimizer.step()
        losses.append(float(loss.detach()))
    return weights.detach(), tuple(losses)


def run_locality_cegis() -> dict[str, object]:
    seed_examples = ((1.0, 1.0),)
    fixed_weights, fixed_losses = _fit_polynomial(seed_examples)
    search_space = (1.0, 2.0, 3.0, -1.0)

    def failures(weights: Tensor) -> tuple[tuple[float, float], ...]:
        found = []
        for value in search_space:
            prediction = float(weights[0] * value + weights[1] * value * value)
            if abs(prediction - 1.0) > 0.1:
                found.append((value, 1.0))
        return tuple(found)

    working = list(seed_examples)
    counterexamples: list[tuple[float, float]] = []
    rounds = 0
    weights = fixed_weights
    for round_index in range(1, 6):
        rounds = round_index
        found = tuple(item for item in failures(weights) if item not in working)
        if not found:
            break
        # Add the worst current counterexample and actually recompile.
        worst = max(
            found,
            key=lambda item: abs(float(weights[0] * item[0] + weights[1] * item[0] ** 2) - item[1]),
        )
        working.append(worst)
        counterexamples.append(worst)
        weights, _ = _fit_polynomial(tuple(working))
    final_failures = failures(weights)
    initial_failures = failures(fixed_weights)
    # Benchmark completion is distinct from the research hypothesis outcome.
    # This organism is intentionally retained as a negative CEGIS result.
    success = bool(counterexamples) and rounds > 1 and bool(fixed_losses)
    return {
        "schema_version": 1,
        "suite": "ModelPactBench",
        "benchmark": "Locality and CEGIS",
        "model": "executed_polynomial_pytorch_organism",
        "initial_validation_error": abs(float(fixed_weights.sum()) - 1.0),
        "initial_search_failures": len(initial_failures),
        "counterexamples_added": [list(item) for item in counterexamples],
        "cegis_rounds": rounds,
        "post_cegis_search_failures": len(final_failures),
        "search_failures_reduced": len(final_failures) < len(initial_failures),
        "status": "PASS" if success else "FAIL",
        "success": success,
        "fixed_probe_steps": len(fixed_losses),
        "negative_result": len(final_failures) > 0,
    }
