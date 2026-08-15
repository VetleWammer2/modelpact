"""Contract execution engine with explicit model and probe-provider boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

from modelpact.contracts.assertions import (
    AssertionEvaluation,
    EvaluationRecord,
    evaluate_assertion,
)
from modelpact.contracts.ast import AssertionType, BehaviorContract, VerificationAssertion
from modelpact.contracts.holdout import (
    HoldoutCapability,
    HoldoutRole,
    SealedHoldoutGate,
)
from modelpact.status import VerificationOutcome
from modelpact.util.hashing import hash_canonical
from modelpact.verify.generation import FreeGenerationRecord


class VerificationRole(StrEnum):
    TARGET = "target"
    GUARD = "guard"
    HOLDOUT_TARGET = "holdout_target"
    HOLDOUT_GUARD = "holdout_guard"


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    adapter_id: str
    base_signature: str
    tokenizer_hash: str
    architecture_hash: str | None = None
    state_schema_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.adapter_id or not self.base_signature or not self.tokenizer_hash:
            raise ValueError("execution identity fields must not be empty")

    def to_dict(self) -> dict[str, str]:
        result = {
            "adapter_id": self.adapter_id,
            "base_signature": self.base_signature,
            "tokenizer_hash": self.tokenizer_hash,
        }
        if self.architecture_hash is not None:
            result["architecture_hash"] = self.architecture_hash
        if self.state_schema_hash is not None:
            result["state_schema_hash"] = self.state_schema_hash
        return result


class AssertionRecordProvider(Protocol):
    def records_for(
        self,
        assertion: VerificationAssertion,
        *,
        source: str,
        role: VerificationRole,
        holdout_capability: HoldoutCapability | None,
    ) -> Sequence[EvaluationRecord]: ...


class UnsupportedRecordProviderError(RuntimeError):
    """The selected adapter cannot produce evidence required by an assertion."""


@runtime_checkable
class GenerationEvidenceProvider(Protocol):
    def generation_evidence(self) -> Sequence[FreeGenerationRecord]: ...


@runtime_checkable
class ProbeHashProvider(Protocol):
    @property
    def probe_hashes(self) -> Mapping[str, str]: ...


@dataclass(slots=True)
class MappingRecordProvider:
    """Small deterministic provider useful for local adapters and tests."""

    records: Mapping[str, Sequence[EvaluationRecord]]

    def records_for(
        self,
        assertion: VerificationAssertion,
        *,
        source: str,
        role: VerificationRole,
        holdout_capability: HoldoutCapability | None,
    ) -> Sequence[EvaluationRecord]:
        del assertion, role, holdout_capability
        return tuple(self.records.get(source, ()))


@dataclass(frozen=True, slots=True)
class VerificationReport:
    schema_version: int
    contract_id: str
    contract_hash: str
    identity: ExecutionIdentity
    outcome: VerificationOutcome
    target_results: tuple[AssertionEvaluation, ...]
    guard_results: tuple[AssertionEvaluation, ...]
    holdout_target_results: tuple[AssertionEvaluation, ...]
    holdout_guard_results: tuple[AssertionEvaluation, ...]
    holdout_outcome: VerificationOutcome
    free_generation_records: tuple[FreeGenerationRecord, ...]
    probe_hashes: Mapping[str, str]
    compatibility_errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only Verification Report schema_version 1 is supported")
        identifiers = [
            result.assertion_id
            for result in (
                *self.target_results,
                *self.guard_results,
                *self.holdout_target_results,
                *self.holdout_guard_results,
            )
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("verification report contains duplicate assertion result IDs")
        if self.outcome is VerificationOutcome.PASS and any(
            result.outcome is not VerificationOutcome.PASS
            for result in (*self.target_results, *self.guard_results)
        ):
            raise ValueError("PASS report contains a non-passing validation assertion")

    @property
    def result_hash(self) -> str:
        return hash_canonical(self.to_dict())

    @property
    def prompt_failures(self) -> tuple[object, ...]:
        return tuple(
            item
            for result in (
                *self.target_results,
                *self.guard_results,
                *self.holdout_target_results,
                *self.holdout_guard_results,
            )
            for item in result.prompt_metrics
            if item.outcome is not VerificationOutcome.PASS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "identity": self.identity.to_dict(),
            "outcome": self.outcome.value,
            "target_results": [result.to_dict() for result in self.target_results],
            "guard_results": [result.to_dict() for result in self.guard_results],
            "holdout_target_results": [result.to_dict() for result in self.holdout_target_results],
            "holdout_guard_results": [result.to_dict() for result in self.holdout_guard_results],
            "holdout_outcome": self.holdout_outcome.value,
            "free_generation_records": [
                record.to_dict() for record in self.free_generation_records
            ],
            "probe_hashes": dict(sorted(self.probe_hashes.items())),
            "compatibility_errors": list(self.compatibility_errors),
            "warnings": list(self.warnings),
            "unsupported_claims": list(self.unsupported_claims),
        }


_GENERATION_TYPES = frozenset(
    {
        AssertionType.EXACT_MATCH,
        AssertionType.NORMALIZED_EXACT_MATCH,
        AssertionType.REGULAR_EXPRESSION,
        AssertionType.JSON_PARSE,
        AssertionType.JSON_SCHEMA,
        AssertionType.FREE_GENERATION_MATCH,
        AssertionType.GENERATION_LENGTH,
    }
)


def _compatibility(contract: BehaviorContract, identity: ExecutionIdentity) -> tuple[str, ...]:
    required = contract.model_requirements
    errors: list[str] = []
    comparisons = (
        ("adapter_id", required.adapter_id, identity.adapter_id),
        ("base_signature", required.base_signature, identity.base_signature),
        ("tokenizer_hash", required.tokenizer_hash, identity.tokenizer_hash),
        ("architecture_hash", required.architecture_hash, identity.architecture_hash),
        ("state_schema_hash", required.state_schema_hash, identity.state_schema_hash),
    )
    for name, expected, observed in comparisons:
        if expected is not None and expected != observed:
            errors.append(f"{name} mismatch: required {expected}, observed {observed}")
    return tuple(errors)


def combine_outcomes(outcomes: Sequence[VerificationOutcome]) -> VerificationOutcome:
    """Combine evidence without treating missing/unsupported work as success."""

    if not outcomes:
        return VerificationOutcome.NOT_APPLICABLE
    if VerificationOutcome.FAIL in outcomes:
        return VerificationOutcome.FAIL
    if VerificationOutcome.UNSUPPORTED in outcomes:
        return VerificationOutcome.UNSUPPORTED
    if VerificationOutcome.INCONCLUSIVE in outcomes:
        return VerificationOutcome.INCONCLUSIVE
    applicable = [item for item in outcomes if item is not VerificationOutcome.NOT_APPLICABLE]
    return VerificationOutcome.PASS if applicable else VerificationOutcome.NOT_APPLICABLE


def _provider_failure(
    assertion: VerificationAssertion,
    message: str,
    *,
    outcome: VerificationOutcome = VerificationOutcome.INCONCLUSIVE,
) -> AssertionEvaluation:
    return AssertionEvaluation(
        assertion_id=assertion.id,
        assertion_type=assertion.type,
        outcome=outcome,
        metric=assertion.type.value,
        value=None,
        margin=None,
        prompt_metrics=(),
        message=message,
    )


def _execute_group(
    assertions: Sequence[VerificationAssertion],
    *,
    role: VerificationRole,
    source_override: str | None,
    provider: AssertionRecordProvider,
    contract: BehaviorContract,
    capability: HoldoutCapability | None,
    schemas: Mapping[str, Mapping[str, object]],
    id_suffix: str = "",
) -> tuple[AssertionEvaluation, ...]:
    results: list[AssertionEvaluation] = []
    for assertion in assertions:
        execution_assertion = (
            replace(assertion, id=f"{assertion.id}{id_suffix}") if id_suffix else assertion
        )
        source = source_override or assertion.source
        try:
            records = provider.records_for(
                assertion,
                source=source,
                role=role,
                holdout_capability=capability,
            )
            result = evaluate_assertion(
                execution_assertion,
                tuple(records),
                statistics=contract.statistics,
                schemas=schemas,
            )
        except UnsupportedRecordProviderError as error:
            result = _provider_failure(
                execution_assertion,
                str(error),
                outcome=VerificationOutcome.UNSUPPORTED,
            )
        except Exception as error:  # trusted adapter failure becomes evidence, never success
            result = _provider_failure(
                execution_assertion,
                f"record provider failed with {type(error).__name__}: {error}",
            )
        results.append(result)
    return tuple(results)


def verify_contract(
    contract: BehaviorContract,
    *,
    identity: ExecutionIdentity,
    provider: AssertionRecordProvider,
    schemas: Mapping[str, Mapping[str, object]] | None = None,
    free_generation_records: Sequence[FreeGenerationRecord] = (),
    probe_hashes: Mapping[str, str] | None = None,
    include_holdout: bool = False,
    holdout_gate: SealedHoldoutGate | None = None,
    holdout_capability: HoldoutCapability | None = None,
) -> VerificationReport:
    """Execute target, guard, free-generation, and optionally holdout evidence."""

    compatibility_errors = _compatibility(contract, identity)
    schema_map = schemas or {}
    target_results = _execute_group(
        contract.targets,
        role=VerificationRole.TARGET,
        source_override=None,
        provider=provider,
        contract=contract,
        capability=None,
        schemas=schema_map,
    )
    guard_results = _execute_group(
        contract.guards,
        role=VerificationRole.GUARD,
        source_override=None,
        provider=provider,
        contract=contract,
        capability=None,
        schemas=schema_map,
    )
    holdout_targets: tuple[AssertionEvaluation, ...] = ()
    holdout_guards: tuple[AssertionEvaluation, ...] = ()
    warnings: list[str] = []
    if include_holdout:
        if holdout_gate is None or holdout_capability is None:
            raise ValueError("holdout execution requires a gate and capability")
        if contract.holdout.targets is not None:
            source = holdout_gate.validate(holdout_capability, HoldoutRole.TARGETS)
            holdout_targets = _execute_group(
                contract.targets,
                role=VerificationRole.HOLDOUT_TARGET,
                source_override=source,
                provider=provider,
                contract=contract,
                capability=holdout_capability,
                schemas=schema_map,
                id_suffix=":holdout-target",
            )
        if contract.holdout.guards is not None:
            source = holdout_gate.validate(holdout_capability, HoldoutRole.GUARDS)
            holdout_guards = _execute_group(
                contract.guards,
                role=VerificationRole.HOLDOUT_GUARD,
                source_override=source,
                provider=provider,
                contract=contract,
                capability=holdout_capability,
                schemas=schema_map,
                id_suffix=":holdout-guard",
            )
    elif contract.holdout.configured:
        warnings.append("sealed holdout was not executed")
    validation_outcomes = [result.outcome for result in (*target_results, *guard_results)]
    outcome = combine_outcomes(validation_outcomes)
    if compatibility_errors:
        outcome = VerificationOutcome.FAIL
    effective_generation_records = tuple(free_generation_records)
    if not effective_generation_records and isinstance(provider, GenerationEvidenceProvider):
        effective_generation_records = tuple(provider.generation_evidence())
    generative_assertions = [
        assertion
        for assertion in (*contract.targets, *contract.guards)
        if assertion.type in _GENERATION_TYPES
    ]
    if generative_assertions and not effective_generation_records:
        warnings.append("generative assertions lack independently recorded generation evidence")
        if outcome is VerificationOutcome.PASS:
            outcome = VerificationOutcome.INCONCLUSIVE
    holdout_outcome = combine_outcomes(
        [result.outcome for result in (*holdout_targets, *holdout_guards)]
    )
    unsupported_claims: list[str] = []
    if not contract.guards:
        unsupported_claims.append("PRESERVATION_ASSERTIONS_VERIFIED")
    if not include_holdout or holdout_outcome is not VerificationOutcome.PASS:
        unsupported_claims.append("SEALED_HOLDOUT_VERIFIED")
    if generative_assertions and not effective_generation_records:
        unsupported_claims.append("FREE_GENERATION_VERIFIED")
    effective_probe_hashes = dict(sorted((probe_hashes or {}).items()))
    if not effective_probe_hashes and isinstance(provider, ProbeHashProvider):
        effective_probe_hashes = dict(sorted(provider.probe_hashes.items()))
    return VerificationReport(
        schema_version=1,
        contract_id=contract.id,
        contract_hash=contract.contract_id,
        identity=identity,
        outcome=outcome,
        target_results=target_results,
        guard_results=guard_results,
        holdout_target_results=holdout_targets,
        holdout_guard_results=holdout_guards,
        holdout_outcome=holdout_outcome,
        free_generation_records=effective_generation_records,
        probe_hashes=effective_probe_hashes,
        compatibility_errors=compatibility_errors,
        warnings=tuple(warnings),
        unsupported_claims=tuple(sorted(unsupported_claims)),
    )


__all__ = [
    "AssertionRecordProvider",
    "ExecutionIdentity",
    "GenerationEvidenceProvider",
    "MappingRecordProvider",
    "ProbeHashProvider",
    "UnsupportedRecordProviderError",
    "VerificationReport",
    "VerificationRole",
    "combine_outcomes",
    "verify_contract",
]
