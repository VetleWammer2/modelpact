"""Additive patch resolution followed by executed contract-closure testing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import torch

from modelpact.compose.contradiction import ContradictionChecker, ContradictionWitness
from modelpact.compose.interactions import contract_margin_interaction
from modelpact.status import CompositionClaim, VerificationOutcome


class MarginKind(StrEnum):
    TARGET = "target"
    GUARD = "guard"
    FREE_GENERATION = "free_generation"


@dataclass(frozen=True, slots=True)
class ContractMargin:
    contract_id: str
    kind: MarginKind
    margin: float
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("contract_id must not be empty")
        if not math.isfinite(self.margin):
            raise ValueError("contract margin must be finite")

    @property
    def passed(self) -> bool:
        return self.margin >= 0.0


@dataclass(frozen=True, slots=True)
class VerificationReport:
    outcome: VerificationOutcome
    margins: tuple[ContractMargin, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifiers = [margin.contract_id for margin in self.margins]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("verification report contains duplicate contract margins")
        if self.outcome is VerificationOutcome.PASS and any(
            not margin.passed for margin in self.margins
        ):
            raise ValueError("PASS report cannot contain a failing contract margin")

    def by_contract(self) -> dict[str, ContractMargin]:
        return {margin.contract_id: margin for margin in self.margins}


@dataclass(frozen=True, slots=True)
class PatchOperand:
    patch_id: str
    base_signature: str
    module_schema_hash: str
    delta: Mapping[str, torch.Tensor]
    contract_ids: tuple[str, ...]
    verified_margins: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.patch_id or not self.base_signature or not self.module_schema_hash:
            raise ValueError("patch identity and compatibility fields must not be empty")
        if not self.contract_ids:
            raise ValueError("a behavior patch must declare at least one contract")
        if len(self.contract_ids) != len(set(self.contract_ids)):
            raise ValueError("patch contract identities must be unique")


class CompositionExecutor(Protocol):
    def __call__(
        self, delta: Mapping[str, torch.Tensor], contract_ids: tuple[str, ...]
    ) -> VerificationReport: ...


@dataclass(frozen=True, slots=True)
class CompositionResult:
    claim: CompositionClaim
    patch_ids: tuple[str, ...]
    contract_ids: tuple[str, ...]
    resolved_delta: Mapping[str, torch.Tensor]
    verification: VerificationReport | None
    contradictions: tuple[ContradictionWitness, ...] = ()
    structural_errors: tuple[str, ...] = ()
    degraded_contracts: tuple[str, ...] = ()
    unverified_contracts: tuple[str, ...] = ()
    interaction_margins: Mapping[str, float] = field(default_factory=dict)
    base_verification: VerificationReport | None = None
    singleton_verifications: Mapping[str, VerificationReport] = field(default_factory=dict)
    degradation_tolerance: float | None = None
    evidence_gaps: tuple[str, ...] = ()

    @property
    def closed(self) -> bool:
        return self.claim is CompositionClaim.COMPOSITION_CLOSED


def _canonical_target(name: str, aliases: Mapping[str, str]) -> str:
    seen: set[str] = set()
    current = name
    while current in aliases:
        if current in seen:
            raise ValueError(f"alias cycle contains {current!r}")
        seen.add(current)
        current = aliases[current]
    return current


def _canonicalize_operand(
    delta: Mapping[str, torch.Tensor], aliases: Mapping[str, str]
) -> dict[str, torch.Tensor]:
    canonical: dict[str, torch.Tensor] = {}
    for name in sorted(delta):
        value = delta[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"delta target {name!r} is not a tensor")
        target = _canonical_target(name, aliases)
        if target in canonical:
            prior = canonical[target]
            incompatible_alias = (
                prior.shape != value.shape
                or prior.dtype != value.dtype
                or not torch.equal(prior, value)
            )
            if incompatible_alias:
                raise ValueError(
                    f"patch provides inconsistent deltas for aliased target {target!r}"
                )
            continue
        canonical[target] = value
    return canonical


def additive_compose(
    operands: tuple[PatchOperand, ...] | list[PatchOperand],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Resolve additive deltas deterministically without mutating operands.

    Patch IDs define accumulation order so the result is independent of the
    declarative stack order.  Aliased keys within one patch must carry identical
    deltas and are applied once to their canonical physical tensor.
    """

    if not operands:
        return {}
    identifiers = [operand.patch_id for operand in operands]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("a patch may occur only once in an additive composition")
    alias_map = aliases or {}
    resolved: dict[str, torch.Tensor] = {}
    for operand in sorted(operands, key=lambda item: item.patch_id):
        canonical = _canonicalize_operand(operand.delta, alias_map)
        for target in sorted(canonical):
            value = canonical[target].detach()
            if target not in resolved:
                resolved[target] = value.clone()
                continue
            prior = resolved[target]
            if prior.shape != value.shape:
                raise ValueError(f"delta shape mismatch for target {target!r}")
            if prior.dtype != value.dtype:
                raise ValueError(f"delta dtype mismatch for target {target!r}")
            if prior.device != value.device:
                raise ValueError(f"delta device mismatch for target {target!r}")
            resolved[target] = prior + value
    return {target: resolved[target] for target in sorted(resolved)}


def _structural_errors(operands: tuple[PatchOperand, ...]) -> tuple[str, ...]:
    if not operands:
        return ("at least one patch is required",)
    errors: list[str] = []
    bases = {operand.base_signature for operand in operands}
    schemas = {operand.module_schema_hash for operand in operands}
    if len(bases) != 1:
        errors.append("patches declare different base signatures")
    if len(schemas) != 1:
        errors.append("patches declare different module schemas")
    identifiers = [operand.patch_id for operand in operands]
    if len(identifiers) != len(set(identifiers)):
        errors.append("patch stack contains duplicate patch identities")
    return tuple(errors)


def verify_contract_closure(
    operands: tuple[PatchOperand, ...] | list[PatchOperand],
    *,
    executor: CompositionExecutor,
    aliases: Mapping[str, str] | None = None,
    contradiction_checker: ContradictionChecker | None = None,
    degradation_tolerance: float | None = None,
    base_margins: Mapping[str, float] | None = None,
    execute_baselines: bool = False,
) -> CompositionResult:
    """Add patches and execute their union contracts.

    ``degradation_tolerance`` classifies still-passing but materially reduced
    margins as ``COMPOSITION_DEGRADED``.  With ``execute_baselines``, the same
    union contract set is executed on the empty stack and every singleton; this
    supplies independent baselines and exact pair-interaction margins. Failed,
    inconclusive, or unsupported composed execution is a semantic conflict,
    never a successful closure claim.
    """

    if degradation_tolerance is not None and (
        not math.isfinite(degradation_tolerance) or degradation_tolerance < 0
    ):
        raise ValueError("degradation_tolerance must be finite and non-negative")
    operand_tuple = tuple(operands)
    patch_ids = tuple(sorted(operand.patch_id for operand in operand_tuple))
    contract_ids = tuple(
        sorted({item for operand in operand_tuple for item in operand.contract_ids})
    )
    errors = _structural_errors(operand_tuple)
    if errors:
        return CompositionResult(
            claim=CompositionClaim.STRUCTURAL_INCOMPATIBILITY,
            patch_ids=patch_ids,
            contract_ids=contract_ids,
            resolved_delta={},
            verification=None,
            structural_errors=errors,
            degradation_tolerance=degradation_tolerance,
        )
    contradictions = contradiction_checker(contract_ids) if contradiction_checker else ()
    if contradictions:
        return CompositionResult(
            claim=CompositionClaim.STATIC_CONTRACT_CONTRADICTION,
            patch_ids=patch_ids,
            contract_ids=contract_ids,
            resolved_delta={},
            verification=None,
            contradictions=tuple(contradictions),
            degradation_tolerance=degradation_tolerance,
        )
    try:
        resolved = additive_compose(operand_tuple, aliases=aliases)
    except (TypeError, ValueError) as error:
        return CompositionResult(
            claim=CompositionClaim.STRUCTURAL_INCOMPATIBILITY,
            patch_ids=patch_ids,
            contract_ids=contract_ids,
            resolved_delta={},
            verification=None,
            structural_errors=(str(error),),
            degradation_tolerance=degradation_tolerance,
        )

    base_verification: VerificationReport | None = None
    singleton_verifications: dict[str, VerificationReport] = {}
    if execute_baselines:
        # Execute all union contracts at the empty stack and at every singleton.
        # A singleton will ordinarily fail contracts owned only by another patch;
        # those signed margins are still necessary for semantic-interaction evidence.
        base_verification = executor({}, contract_ids)
        for operand in sorted(operand_tuple, key=lambda item: item.patch_id):
            singleton_delta = additive_compose((operand,), aliases=aliases)
            singleton_verifications[operand.patch_id] = executor(singleton_delta, contract_ids)
    report = executor(resolved, contract_ids)
    reported_contracts = set(report.by_contract())
    missing_contracts = tuple(sorted(set(contract_ids) - reported_contracts))
    if (
        report.outcome is not VerificationOutcome.PASS
        or missing_contracts
        or any(not margin.passed for margin in report.margins)
    ):
        claim = CompositionClaim.SEMANTIC_CONFLICT
    else:
        claim = CompositionClaim.COMPOSITION_CLOSED

    degraded: list[str] = []
    if claim is CompositionClaim.COMPOSITION_CLOSED and degradation_tolerance is not None:
        parent_baselines: dict[str, float] = {}
        for operand in operand_tuple:
            executed = singleton_verifications.get(operand.patch_id)
            executed_margins = executed.by_contract() if executed is not None else {}
            for contract_id in operand.contract_ids:
                margin = executed_margins.get(contract_id)
                operand_margin = (
                    margin.margin
                    if margin is not None
                    else operand.verified_margins.get(contract_id)
                )
                if operand_margin is None:
                    continue
                parent_baselines[contract_id] = min(
                    parent_baselines.get(contract_id, operand_margin), operand_margin
                )
        for contract_margin in report.margins:
            baseline = parent_baselines.get(contract_margin.contract_id)
            if baseline is not None and baseline - contract_margin.margin > degradation_tolerance:
                degraded.append(contract_margin.contract_id)
        if degraded:
            claim = CompositionClaim.COMPOSITION_DEGRADED

    interactions: dict[str, float] = {}
    evidence_gaps: list[str] = []
    if len(operand_tuple) == 2:
        final_margins = report.by_contract()
        ordered_operands = tuple(sorted(operand_tuple, key=lambda item: item.patch_id))
        left_executed = singleton_verifications.get(ordered_operands[0].patch_id)
        right_executed = singleton_verifications.get(ordered_operands[1].patch_id)
        left_executed_margins = left_executed.by_contract() if left_executed is not None else {}
        right_executed_margins = right_executed.by_contract() if right_executed is not None else {}
        base_executed_margins = (
            base_verification.by_contract() if base_verification is not None else {}
        )
        for contract_id in sorted(final_margins):
            left_margin = left_executed_margins.get(contract_id)
            right_margin = right_executed_margins.get(contract_id)
            base_margin = base_executed_margins.get(contract_id)
            left = (
                left_margin.margin
                if left_margin is not None
                else ordered_operands[0].verified_margins.get(contract_id)
            )
            right = (
                right_margin.margin
                if right_margin is not None
                else ordered_operands[1].verified_margins.get(contract_id)
            )
            base = (
                base_margin.margin
                if base_margin is not None
                else (base_margins or {}).get(contract_id)
            )
            if left is None or right is None or base is None:
                evidence_gaps.append(
                    f"contract-margin interaction unavailable for {contract_id}: "
                    "base and both singleton margins are required"
                )
                continue
            interactions[contract_id] = contract_margin_interaction(
                base_margin=base,
                left_margin=left,
                right_margin=right,
                composed_margin=final_margins[contract_id].margin,
            )
    return CompositionResult(
        claim=claim,
        patch_ids=patch_ids,
        contract_ids=contract_ids,
        resolved_delta=resolved,
        verification=report,
        degraded_contracts=tuple(sorted(degraded)),
        unverified_contracts=missing_contracts,
        interaction_margins=interactions,
        base_verification=base_verification,
        singleton_verifications=dict(sorted(singleton_verifications.items())),
        degradation_tolerance=degradation_tolerance,
        evidence_gaps=tuple(sorted(evidence_gaps)),
    )
