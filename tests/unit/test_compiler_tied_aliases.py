from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn

from modelpact.adapters.tiny_lm import TinyCausalLM, TinyConfig
from modelpact.compiler.constraints import DifferentiableObjective
from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
from modelpact.compiler.package import compilation_delta_program
from modelpact.models.aliases import discover_parameter_aliases
from modelpact.models.schema import inspect_state_schema
from modelpact.patch.ast import Alias
from modelpact.patch.mount import mount_patch


def _tiny_tied_model() -> TinyCausalLM:
    return TinyCausalLM(
        TinyConfig(
            max_sequence_length=16,
            hidden_size=8,
            intermediate_size=12,
            num_layers=1,
            num_heads=2,
            tie_word_embeddings=True,
            initialization_seed=19,
        )
    )


def _tied_objective(model: nn.Module, target: Tensor) -> Tensor:
    assert isinstance(model, TinyCausalLM)
    # Both access paths participate in the optimization graph.  A compiler that
    # patches only lm_head learns a materially different delta than runtime,
    # where the patch program applies the same delta to both aliases.
    value = model.token_embedding.weight[7, 0] + model.lm_head.weight[7, 0]
    return (value - target.to(value.device)) ** 2


def test_tied_alias_optimization_matches_packaged_runtime_semantics() -> None:
    base = _tiny_tied_model()
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    groups = discover_parameter_aliases(base)
    assert any(group.members == ("lm_head.weight", "token_embedding.weight") for group in groups)
    target = torch.tensor(0.75)
    result = compile_low_rank_patch(
        base,
        (DifferentiableObjective("tied-target", (target,), _tied_objective),),
        (),
        config=OptimizerConfig(
            maximum_rank=1,
            maximum_modules=1,
            steps=160,
            learning_rate=0.08,
            patience=80,
            complexity_weight=1e-8,
            seed=23,
        ),
    )
    assert result.feasible
    assert result.active_modules == ("lm_head",)
    assert result.metadata["optimized_aliases"] == {
        "lm_head": ["lm_head.weight", "token_embedding.weight"]
    }
    # The scalar factor parameter count is r(m+n). Alias views do not add
    # duplicate optimizer entries.
    left, right = result.factors["lm_head"]
    assert result.metadata["trainable_factor_parameters"] == left.numel() + right.numel()
    assert result.best_target_loss is not None

    schema = inspect_state_schema(base)
    program, tensors = compilation_delta_program(result, schema)
    assert isinstance(program.targets["token_embedding.weight"], Alias)
    assert "lm_head.weight" in program.targets

    runtime = copy.deepcopy(base)
    session = mount_patch(runtime, program, tensors, state_schema=schema)
    try:
        runtime_loss = float(_tied_objective(runtime, target).detach())
        assert runtime_loss == pytest.approx(result.best_target_loss, abs=1e-7, rel=1e-6)

        # A physical folded update changes the one shared Parameter once.  Its
        # forward behavior must equal runtime application through both aliases.
        folded = copy.deepcopy(base)
        delta = result.deltas["lm_head"]
        with torch.no_grad():
            folded.token_embedding.weight.add_(delta)
        input_ids = torch.tensor([[1, 7, 11]], dtype=torch.long)
        with torch.no_grad():
            expected_logits = folded(input_ids).logits
            mounted_logits = runtime(input_ids).logits
        torch.testing.assert_close(mounted_logits, expected_logits, atol=1e-6, rtol=1e-5)
    finally:
        session.unmount()

    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
