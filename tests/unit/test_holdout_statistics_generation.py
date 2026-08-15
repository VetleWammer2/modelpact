from __future__ import annotations

from pathlib import Path

import pytest

from modelpact.contracts import (
    HoldoutAccessError,
    HoldoutCapability,
    HoldoutPhase,
    HoldoutRole,
    SealedHoldoutGate,
    adjust_p_values,
    bootstrap_mean_interval,
    loads_contract,
    paired_bootstrap,
)
from modelpact.contracts.ast import GenerationPolicy
from modelpact.verify.generation import (
    GeneratedOutput,
    GenerationRequest,
    execute_free_generation,
)


def sealed_contract() -> object:
    return loads_contract(
        """
schema_version: 1
id: holdout-test
model_requirements: {output_semantics: causal_lm}
compile: {objectives: []}
verify:
  targets:
    - {id: exact, type: exact_match, source: validation.jsonl, expected: ok}
  guards: []
holdout:
  sealed: true
  targets: sealed/targets.jsonl
  unseal_policy: final_candidate_only
statistics: {bootstrap_samples: 10, bootstrap_seed: 2}
generation: {mode: greedy, max_new_tokens: 2}
"""
    )


def test_holdout_rejects_compile_search_and_validation_access() -> None:
    gate = SealedHoldoutGate(sealed_contract())  # type: ignore[arg-type]
    for phase in (
        HoldoutPhase.COMPILATION,
        HoldoutPhase.COUNTEREXAMPLE_SEARCH,
        HoldoutPhase.VALIDATION,
    ):
        with pytest.raises(HoldoutAccessError, match="inaccessible"):
            gate.authorize(phase=phase, candidate_id="candidate")


def test_holdout_is_one_way_and_capability_bound(tmp_path: Path) -> None:
    contract = sealed_contract()
    gate = SealedHoldoutGate(contract)  # type: ignore[arg-type]
    gate.select_final_candidate("candidate-a")
    with pytest.raises(HoldoutAccessError, match="not selected"):
        gate.authorize(phase=HoldoutPhase.FINAL_CANDIDATE, candidate_id="candidate-b")
    capability = gate.authorize(
        phase=HoldoutPhase.FINAL_CANDIDATE,
        candidate_id="candidate-a",
    )
    (tmp_path / "sealed").mkdir()
    (tmp_path / "sealed" / "targets.jsonl").write_text('{"prompt":"x"}\n')
    assert b'"prompt"' in gate.read_bytes(
        capability,
        role=HoldoutRole.TARGETS,
        contract_root=tmp_path,
    )
    fake = HoldoutCapability(
        contract_hash=capability.contract_hash,
        candidate_id=capability.candidate_id,
        phase=capability.phase,
        nonce="00" * 32,
    )
    with pytest.raises(HoldoutAccessError, match="invalid"):
        gate.validate(fake, HoldoutRole.TARGETS)
    with pytest.raises(HoldoutAccessError, match="consumed"):
        gate.authorize(phase=HoldoutPhase.FINAL_CANDIDATE, candidate_id="candidate-a")
    assert gate.access_records[0].candidate_id == "candidate-a"


def test_paired_bootstrap_is_paired_and_deterministic() -> None:
    first = paired_bootstrap(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
        samples=200,
        seed=7,
    )
    second = paired_bootstrap(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
        samples=200,
        seed=7,
    )
    assert first == second
    assert first.interval.estimate == 1.0
    assert first.interval.lower == first.interval.upper == 1.0
    assert first.probability_greater_than_zero == 1.0


def test_bootstrap_and_multiple_comparison_validation() -> None:
    interval = bootstrap_mean_interval([0.0, 1.0, 1.0], samples=100, seed=4)
    assert interval.lower <= interval.estimate <= interval.upper
    assert adjust_p_values([0.01, 0.04, 0.2], method="bonferroni") == (
        0.03,
        0.12,
        pytest.approx(0.6),
    )
    holm = adjust_p_values([0.01, 0.04, 0.2], method="holm")
    assert holm[0] <= holm[1] <= holm[2]
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap([1.0], [1.0, 2.0])


class _Backend:
    def generate(
        self,
        prompt: str,
        *,
        policy: GenerationPolicy,
        seed: int,
    ) -> GeneratedOutput:
        del policy
        text = f"{prompt}:{seed}"
        return GeneratedOutput(
            text=text,
            token_ids=(seed + 1,),
            token_log_probabilities=(-0.25,),
            parser_result={"parsed": True},
        )


def test_free_generation_records_policy_seed_and_content_hashes() -> None:
    policy = GenerationPolicy(max_new_tokens=2, seeds=(3, 1))
    execution = execute_free_generation(
        _Backend(),
        (GenerationRequest("b", "second"), GenerationRequest("a", "first")),
        policy=policy,
    )
    assert [item.sample_id for item in execution.records] == ["a", "a", "b", "b"]
    assert [item.seed for item in execution.records] == [3, 1, 3, 1]
    assert all(item.prompt_hash.startswith("sha256:") for item in execution.records)
    assert all(item.output_hash.startswith("sha256:") for item in execution.records)
    policy_hash = execution.records[0].generation_policy_hash
    assert all(item.generation_policy_hash == policy_hash for item in execution.records)
    assert execution.records[0].token_diagnostics[0]["log_probability"] == -0.25
