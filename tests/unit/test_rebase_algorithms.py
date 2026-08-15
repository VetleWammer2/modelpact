from __future__ import annotations

from typing import cast

import torch

from modelpact.rebase.compile import (
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
from modelpact.status import RebaseClaim, VerificationOutcome


def _base(
    signature: str,
    *,
    architecture: str = "tiny-a",
    schema: str = "schema-a",
    tokenizer: str = "tokenizer",
    shapes: dict[str, tuple[int, ...]] | None = None,
) -> BaseModelDescriptor:
    return BaseModelDescriptor(
        signature=signature,
        architecture_id=architecture,
        module_schema_hash=schema,
        tokenizer_hash=tokenizer,
        output_semantics="causal_lm",
        module_shapes=shapes or {"weight": (1,)},
        family_id="tiny",
    )


def _patch() -> RebasePatch:
    return RebasePatch(
        patch_id="patch-v1",
        source_base_signature="base-v1",
        delta={"weight": torch.tensor([1.0])},
        target_contract_ids=("target",),
        preservation_contract_ids=("old-guard",),
    )


def _passing_verification() -> RebaseVerification:
    return RebaseVerification(
        VerificationOutcome.PASS,
        target_margins={"target": 0.5},
        guard_margins={"new-guard": 0.2, "old-guard": 0.4},
    )


def test_direct_transfer_is_verified_and_skips_recompiler() -> None:
    request = RebaseRequest(
        patch=_patch(),
        source_base=_base("base-v1"),
        target_base=_base("base-v2"),
        new_base_guard_ids=("new-guard",),
        budget=RebaseBudget(10),
    )

    def unexpected(_request: object) -> object:
        raise AssertionError("semantic recompilation must not run after verified direct transfer")

    result = semantic_rebase(
        request,
        applier=lambda delta, _target: delta,
        verifier=lambda _candidate, _targets, _guards: _passing_verification(),
        teacher_builder=unexpected,  # type: ignore[arg-type]
        recompiler=unexpected,  # type: ignore[arg-type]
    )
    assert result.disposition is RebaseDisposition.DIRECT_TRANSPLANT_VERIFIED
    assert result.claim is RebaseClaim.DIRECT_TRANSPLANT_VERIFIED
    assert result.direct_transfer.attempted
    assert result.direct_transfer.verified


def test_failed_direct_transfer_triggers_behavioral_recompile() -> None:
    calls = {"teacher": 0, "compiler": 0, "verify": 0}
    request = RebaseRequest(
        patch=_patch(),
        source_base=_base("base-v1"),
        target_base=_base("base-v2"),
        new_base_guard_ids=("new-guard",),
        budget=RebaseBudget(20),
    )

    def verify(
        candidate: object, _targets: tuple[str, ...], _guards: tuple[str, ...]
    ) -> RebaseVerification:
        calls["verify"] += 1
        delta = cast(dict[str, torch.Tensor], candidate)
        if float(delta["weight"].item()) == 1.0:
            return RebaseVerification(
                VerificationOutcome.FAIL,
                target_margins={"target": -0.5},
                guard_margins={"new-guard": 0.3, "old-guard": 0.3},
            )
        return _passing_verification()

    def teachers(_request: RebaseRequest) -> TeacherContext:
        calls["teacher"] += 1
        return TeacherContext("old-patched", "new-base", {"target": 0.8}, evidence_count=4)

    def recompile(recompile_request: object) -> BehavioralRecompileResult:
        calls["compiler"] += 1
        assert recompile_request.old_patched_teacher == "old-patched"
        assert recompile_request.new_unpatched_teacher == "new-base"
        assert recompile_request.direct_transfer.attempted
        return BehavioralRecompileResult(
            candidate_delta={"weight": torch.tensor([0.25])},
            optimization_succeeded=True,
            budget_exhausted=False,
            steps_executed=11,
            restarts_executed=1,
            complexity={"parameters": 1},
        )

    result = semantic_rebase(
        request,
        applier=lambda delta, _target: delta,
        verifier=verify,
        teacher_builder=teachers,
        recompiler=recompile,
    )
    assert result.claim is RebaseClaim.SEMANTIC_REBASE_VERIFIED
    assert result.disposition is RebaseDisposition.SEMANTIC_REBASE_VERIFIED
    assert calls == {"teacher": 1, "compiler": 1, "verify": 2}
    assert result.evidence.old_patched_behavior == {"target": 0.8}


def test_cross_architecture_rebase_never_directly_transplants_tensors() -> None:
    application_values: list[float] = []
    request = RebaseRequest(
        patch=_patch(),
        source_base=_base("base-v1"),
        target_base=_base(
            "base-v2",
            architecture="tiny-b",
            schema="schema-b",
            shapes={"different.weight": (2,)},
        ),
        new_base_guard_ids=("new-guard",),
        budget=RebaseBudget(10),
    )

    def apply(delta: dict[str, torch.Tensor] | object, _target: BaseModelDescriptor) -> object:
        mapped = cast(dict[str, torch.Tensor], delta)
        application_values.append(float(mapped["different.weight"].sum().item()))
        return mapped

    result = semantic_rebase(
        request,
        applier=apply,
        verifier=lambda _candidate, _targets, _guards: _passing_verification(),
        teacher_builder=lambda _request: TeacherContext("old", "new", {"target": 0.4}, 3),
        recompiler=lambda _request: BehavioralRecompileResult(
            candidate_delta={"different.weight": torch.tensor([0.1, 0.2])},
            optimization_succeeded=True,
            budget_exhausted=False,
            steps_executed=4,
            restarts_executed=1,
        ),
    )
    assert not result.direct_transfer.attempted
    assert result.claim is RebaseClaim.SEMANTIC_REBASE_VERIFIED
    assert application_values == [float(torch.tensor(0.3).item())]


def test_incompatible_tokenizer_is_inconclusive_and_does_not_compile() -> None:
    request = RebaseRequest(
        patch=_patch(),
        source_base=_base("base-v1"),
        target_base=_base("base-v2", tokenizer="different"),
        new_base_guard_ids=("new-guard",),
        budget=RebaseBudget(10),
    )

    def unexpected(_request: object) -> object:
        raise AssertionError("incompatible semantics must stop before compiler execution")

    result = semantic_rebase(
        request,
        applier=lambda delta, _target: delta,
        verifier=lambda _candidate, _targets, _guards: _passing_verification(),
        teacher_builder=unexpected,  # type: ignore[arg-type]
        recompiler=unexpected,  # type: ignore[arg-type]
    )
    assert result.disposition is RebaseDisposition.INCOMPATIBLE_SEMANTICS
    assert result.claim is RebaseClaim.REBASE_INCONCLUSIVE


def test_recompile_budget_failure_is_not_described_as_impossibility() -> None:
    request = RebaseRequest(
        patch=_patch(),
        source_base=_base("base-v1"),
        target_base=_base("base-v2"),
        new_base_guard_ids=("new-guard",),
        budget=RebaseBudget(3),
    )
    direct_fail = RebaseVerification(
        VerificationOutcome.FAIL,
        target_margins={"target": -1.0},
        guard_margins={"new-guard": 0.2, "old-guard": 0.2},
    )
    result = semantic_rebase(
        request,
        applier=lambda delta, _target: delta,
        verifier=lambda _candidate, _targets, _guards: direct_fail,
        teacher_builder=lambda _request: TeacherContext("old", "new", {"target": 0.5}, 2),
        recompiler=lambda _request: BehavioralRecompileResult(
            candidate_delta=None,
            optimization_succeeded=False,
            budget_exhausted=True,
            steps_executed=3,
            restarts_executed=1,
            best_target_margins={"target": -0.1},
            failure_reason="no feasible candidate found within the declared budget",
        ),
    )
    assert result.claim is RebaseClaim.REBASE_FAILED
    assert result.evidence.budget_exhausted
    assert "impossible" not in " ".join(result.evidence.warnings).lower()
