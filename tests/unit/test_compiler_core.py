from __future__ import annotations

import pytest
import torch
from torch import nn

from modelpact.adapters.base import ModelBatch
from modelpact.compiler.analysis import analyze_candidate_modules
from modelpact.compiler.cegis import CEGISStop, Counterexample, run_cegis
from modelpact.compiler.constraints import (
    DifferentiableConstraint,
    DifferentiableObjective,
    MultiplierState,
)
from modelpact.compiler.extract import (
    ExtractionPromptRoles,
    build_extraction_prompt_roles,
    run_extraction_cegis,
)
from modelpact.compiler.gradient_basis import low_rank_factors
from modelpact.compiler.minimize import minimize_patch
from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
from modelpact.compiler.result import CompilationResult, CompilationStatus
from modelpact.diff.witnesses import DifferenceWitness
from modelpact.status import MinimalityClaim


def _loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    inputs, targets = batch
    return torch.nn.functional.mse_loss(model(inputs), targets)


def test_low_rank_initialization_reconstructs_rank_one() -> None:
    matrix = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    left, right = low_rank_factors(matrix, rank=1)
    torch.testing.assert_close(left @ right, matrix, atol=1e-5, rtol=1e-5)


def test_compiler_supports_target_delta_initialization() -> None:
    base = nn.Sequential(nn.Linear(2, 2, bias=False))
    target = nn.Sequential(nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        base[0].weight.zero_()
        target[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    batch = (torch.eye(2), torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    objective = DifferentiableObjective("target", (batch,), _loss)
    result = compile_low_rank_patch(
        base,
        (objective,),
        (),
        config=OptimizerConfig(
            maximum_rank=1,
            maximum_modules=1,
            steps=1,
            learning_rate=1e-3,
            patience=1,
        ),
        target_state=dict(target.named_parameters(remove_duplicate=False)),
    )
    initialization = result.metadata["initialization"]
    assert isinstance(initialization, dict)
    assert initialization["0"]["source"] == "target_delta_signal"


def test_multiplier_updates_only_on_positive_violation() -> None:
    state = MultiplierState(value=1.0, learning_rate=0.5)
    state.update(2.0)
    assert state.value == 2.0
    state.update(-10.0)
    assert state.value == 0.0


def test_candidate_analysis_finds_real_gradient() -> None:
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
    batch = (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]]))
    objective = DifferentiableObjective("target", (batch,), _loss)
    evidence = analyze_candidate_modules(model, (objective,), ())
    assert evidence
    assert evidence[0].target_gradient_norm > 0


def test_compiler_tracks_best_feasible_and_does_not_mutate_base() -> None:
    torch.manual_seed(3)
    base = nn.Sequential(nn.Linear(2, 1, bias=False))
    before = base[0].weight.detach().clone()
    target_batch = (torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0]]))
    guard_batch = (torch.tensor([[0.0, 1.0]]), base(torch.tensor([[0.0, 1.0]])).detach())
    result = compile_low_rank_patch(
        base,
        (DifferentiableObjective("target", (target_batch,), _loss),),
        (DifferentiableConstraint("guard", (guard_batch,), _loss, maximum=1e-4),),
        config=OptimizerConfig(
            maximum_rank=1,
            maximum_modules=1,
            steps=120,
            learning_rate=0.05,
            patience=40,
            seed=9,
        ),
    )
    assert result.feasible
    assert result.best_step is not None
    torch.testing.assert_close(base[0].weight, before)


def test_compiler_never_snapshots_an_unchecked_post_step_candidate() -> None:
    base = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        base[0].weight.zero_()
    inputs = torch.ones(1, 1)
    target_batch = (inputs, torch.tensor([[2.0]]))
    guard_batch = (inputs, torch.zeros(1, 1))

    result = compile_low_rank_patch(
        base,
        (DifferentiableObjective("target", (target_batch,), _loss),),
        (DifferentiableConstraint("guard", (guard_batch,), _loss, maximum=1e-4),),
        config=OptimizerConfig(
            maximum_rank=1,
            maximum_modules=1,
            steps=1,
            learning_rate=0.5,
            patience=1,
            seed=1,
        ),
    )

    assert result.feasible
    patched_weight = base[0].weight.detach() + result.deltas["0"]
    guard_violation = float(patched_weight.square().item()) - 1e-4
    assert guard_violation <= 0
    assert result.evidence[0].guard_margins["guard"] > 0
    assert result.best_step == -1
    assert not result.violated_constraints


def test_cegis_inserts_real_counterexample() -> None:
    calls: list[tuple[str, ...]] = []

    def compile_candidate(targets: tuple[str, ...], guards: tuple[str, ...]) -> CompilationResult:
        calls.append(targets)
        return CompilationResult(CompilationStatus.FEASIBLE, {}, {}, (), {})

    def target_search(candidate: CompilationResult, budget: int) -> tuple[Counterexample[str], ...]:
        if len(calls) == 1:
            return (Counterexample("variant", "target", -1.0, True),)
        return ()

    result = run_cegis(
        ("seed",),
        (),
        compile_candidate=compile_candidate,
        search_targets=target_search,
        search_guards=lambda _candidate, _budget: (),
    )
    assert result.stop_reason is CEGISStop.NO_COUNTEREXAMPLE_WITHIN_BUDGET
    assert result.working_target_examples == ("seed", "variant")
    assert len(calls) == 2


def test_module_minimization_executes_candidates() -> None:
    deltas = {"needed": torch.eye(2), "unused": torch.zeros(2, 2)}
    result = minimize_patch(deltas, lambda candidate: "needed" in candidate)
    assert tuple(result.deltas) == ("needed",)
    assert result.verification_budget_used > 0
    assert result.candidates[0].operation == "verify:initial"
    assert result.candidates[0].passed


def test_minimization_rejects_an_initially_failing_patch() -> None:
    with pytest.raises(ValueError, match="fails its executed verifier"):
        minimize_patch({"delta": torch.eye(2)}, lambda _candidate: False)


def test_minimization_records_factor_rank_and_does_not_overclaim_at_budget() -> None:
    left = torch.ones(16, 1)
    right = torch.ones(1, 16)
    ranked = minimize_patch(
        {"delta": left @ right},
        lambda candidate: "delta" in candidate,
        initial_factors={"delta": (left, right)},
        verification_budget=2,
    )
    assert ranked.candidates[0].ranks == {"delta": 1}

    exhausted = minimize_patch(
        {"a": torch.ones(1, 1), "b": torch.ones(1, 1)},
        lambda _candidate: True,
        verification_budget=2,
    )
    assert MinimalityClaim.MODULE_ONE_MINIMAL not in exhausted.claims
    assert MinimalityClaim.BUDGET_MINIMAL in exhausted.claims


def test_extraction_prompt_roles_are_deterministic_and_globally_disjoint() -> None:
    selected = DifferenceWitness.create(
        original_input="Please transform Aster number 7.",
        minimized_input="Transform Aster number 7.",
        divergence_metrics={"kl": 1.0},
        base_output="base",
        target_output="target",
    )
    guard = DifferenceWitness.create(
        original_input="Please retain Beryl number 13.",
        minimized_input="Retain Beryl number 13.",
        divergence_metrics={"kl": 1.0},
        base_output="base",
        target_output="other-target-change",
    )
    first = build_extraction_prompt_roles(
        (selected,),
        (guard,),
        maximum_rounds=2,
        search_budget_per_domain_per_round=2,
        seed=71,
    )
    second = build_extraction_prompt_roles(
        (selected,),
        (guard,),
        maximum_rounds=2,
        search_budget_per_domain_per_round=2,
        seed=71,
    )
    assert first == second
    groups = (
        first.compile_targets,
        first.compile_guards,
        first.search_targets,
        first.search_guards,
        first.validation_targets,
        first.validation_guards,
        first.holdout_targets,
        first.holdout_guards,
    )
    flattened = [prompt for group in groups for prompt in group]
    assert len(flattened) == len(set(flattened))
    assert "Transform Aster number 7." not in str(first.to_dict())


class _ExtractionTokenizer:
    pad_token_id = 0
    bos_token_id = 0
    eos_token_id = 1
    vocab_size = 3

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        del add_bos, add_eos
        if text == "target-seed":
            return [0]
        if text == "target-search":
            return [1]
        return [2]

    def decode(self, token_ids: object, *, skip_special_tokens: bool = True) -> str:
        del token_ids, skip_special_tokens
        return ""

    def batch(self, texts: object, *, add_bos: bool = True) -> ModelBatch:
        del add_bos
        values = list(texts)  # type: ignore[arg-type]
        ids = torch.tensor([self.encode(str(value)) for value in values], dtype=torch.long)
        return ModelBatch(ids, torch.ones_like(ids, dtype=torch.bool))


class _ExtractionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)
        nn.init.zeros_(self.linear.weight)


class _ExtractionAdapter:
    adapter_id = "tests.extraction"

    def __init__(self) -> None:
        self._tokenizer = _ExtractionTokenizer()

    def tokenizer(self) -> _ExtractionTokenizer:
        return self._tokenizer

    def forward_logits(self, model: nn.Module, batch: ModelBatch) -> torch.Tensor:
        features = torch.nn.functional.one_hot(batch.input_ids, num_classes=3).to(torch.float32)
        return model.linear(features)  # type: ignore[attr-defined, no-any-return]


def test_extraction_cegis_executes_and_inserts_teacher_divergence() -> None:
    base = _ExtractionModel()
    target = _ExtractionModel()
    with torch.no_grad():
        target.linear.weight[:, 0] = torch.tensor([4.0, -4.0])
        target.linear.weight[:, 1] = torch.tensor([4.0, -4.0])
    roles = ExtractionPromptRoles(
        compile_targets=("target-seed",),
        compile_guards=("guard-seed",),
        search_targets=("target-search",),
        search_guards=("guard-search",),
        validation_targets=("target-validation",),
        validation_guards=("guard-validation",),
        holdout_targets=("target-holdout",),
        holdout_guards=("guard-holdout",),
    )
    evidence = run_extraction_cegis(
        _ExtractionAdapter(),  # type: ignore[arg-type]
        base,
        target,
        roles,
        optimizer_config=OptimizerConfig(
            maximum_rank=2,
            maximum_modules=1,
            steps=80,
            learning_rate=0.1,
            patience=40,
            seed=3,
        ),
        maximum_rounds=2,
        search_budget_per_domain_per_round=1,
        maximum_selected_kl=0.05,
        maximum_nonselected_base_kl=0.01,
    )
    assert evidence.compiler_result.feasible
    assert len(evidence.attempts) == 2
    assert evidence.result.working_target_examples == ("target-seed", "target-search")
    assert evidence.result.rounds[0].target_counterexamples
    assert evidence.to_dict()["model_executions"] > 0  # type: ignore[operator]
