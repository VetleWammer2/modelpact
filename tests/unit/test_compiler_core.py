from __future__ import annotations

import torch
from torch import nn

from modelpact.compiler.analysis import analyze_candidate_modules
from modelpact.compiler.cegis import Counterexample, CEGISStop, run_cegis
from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective, MultiplierState
from modelpact.compiler.gradient_basis import low_rank_factors
from modelpact.compiler.minimize import minimize_patch
from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
from modelpact.compiler.result import CompilationResult, CompilationStatus


def _loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    inputs, targets = batch
    return torch.nn.functional.mse_loss(model(inputs), targets)


def test_low_rank_initialization_reconstructs_rank_one() -> None:
    matrix = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    left, right = low_rank_factors(matrix, rank=1)
    torch.testing.assert_close(left @ right, matrix, atol=1e-5, rtol=1e-5)


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
        config=OptimizerConfig(maximum_rank=1, maximum_modules=1, steps=120, learning_rate=0.05, patience=40, seed=9),
    )
    assert result.feasible
    assert result.best_step is not None
    torch.testing.assert_close(base[0].weight, before)


def test_cegis_inserts_real_counterexample() -> None:
    calls: list[tuple[str, ...]] = []

    def compile_candidate(targets: tuple[str, ...], guards: tuple[str, ...]) -> CompilationResult:
        calls.append(targets)
        return CompilationResult(CompilationStatus.FEASIBLE, {}, {}, (), {})

    def target_search(candidate: CompilationResult, budget: int) -> tuple[Counterexample[str], ...]:
        if len(calls) == 1:
            return (Counterexample("variant", "target", -1.0, True),)
        return ()

    result = run_cegis(("seed",), (), compile_candidate=compile_candidate, search_targets=target_search, search_guards=lambda _candidate, _budget: ())
    assert result.stop_reason is CEGISStop.NO_COUNTEREXAMPLE_WITHIN_BUDGET
    assert result.working_target_examples == ("seed", "variant")
    assert len(calls) == 2


def test_module_minimization_executes_candidates() -> None:
    deltas = {"needed": torch.eye(2), "unused": torch.zeros(2, 2)}
    result = minimize_patch(deltas, lambda candidate: "needed" in candidate)
    assert tuple(result.deltas) == ("needed",)
    assert result.verification_budget_used > 0
