"""Conservative static contradiction checks for normalized contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

from modelpact.compose.contradiction import (
    ContradictionWitness,
    StaticRequirement,
    find_static_contradictions,
    incompatible_identity_requirements,
)
from modelpact.contracts.ast import AssertionType, BehaviorContract, VerificationAssertion
from modelpact.util.hashing import sha256_bytes

RecordValue: TypeAlias = str | int | float | bool | None | list[object] | dict[str, object]
ProbeRecord: TypeAlias = Mapping[str, RecordValue]


class StaticCheckStatus(StrEnum):
    CONTRADICTION = "STATIC_CONTRACT_CONTRADICTION"
    NO_CONTRADICTION_FOUND = "NO_STATIC_CONTRADICTION_FOUND"


@dataclass(frozen=True, slots=True)
class StaticCheckResult:
    status: StaticCheckStatus
    witnesses: tuple[ContradictionWitness, ...]
    checked_contracts: tuple[str, ...]
    requirements_examined: int

    @property
    def contradictory(self) -> bool:
        return self.status is StaticCheckStatus.CONTRADICTION

    @property
    def conclusion(self) -> str:
        if self.contradictory:
            return "A contradiction is justified by the recorded static witnesses."
        return "No supported static contradiction was found; satisfiability was not established."


def _prompt_subject(record: ProbeRecord, assertion: VerificationAssertion) -> str | None:
    input_hash = record.get("input_hash", assertion.option("input_hash"))
    if isinstance(input_hash, str) and input_hash:
        return input_hash
    prompt = record.get("prompt", assertion.option("prompt"))
    if isinstance(prompt, str):
        return sha256_bytes(prompt.encode("utf-8"))
    return None


def _records_for(
    assertion: VerificationAssertion,
    records_by_source: Mapping[str, Sequence[ProbeRecord]],
) -> tuple[ProbeRecord, ...]:
    records = records_by_source.get(assertion.source)
    if records is None:
        return ({},)
    return tuple(records)


def _string_option_or_record(
    assertion: VerificationAssertion, record: ProbeRecord, name: str
) -> str | None:
    value = record.get(name, assertion.option(name))
    return value if isinstance(value, str) else None


def _numeric_requirements(
    contract: BehaviorContract,
    assertion: VerificationAssertion,
    subject: str,
) -> list[StaticRequirement]:
    result: list[StaticRequirement] = []
    for option, relation in (("minimum", ">="), ("maximum", "<=")):
        value = assertion.option(option)
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            result.append(
                StaticRequirement(
                    contract_id=contract.id,
                    subject=f"{subject}:{assertion.type.value}",
                    assertion_type="numeric_acceptance",
                    relation=relation,
                    value=value,
                )
            )
    return result


def contract_static_requirements(
    contract: BehaviorContract,
    *,
    records_by_source: Mapping[str, Sequence[ProbeRecord]] | None = None,
) -> tuple[StaticRequirement, ...]:
    """Lower only statically comparable assertion fragments.

    Assertions whose concrete prompt or expected value is stored in an
    unavailable probe file are intentionally skipped.
    """

    records = records_by_source or {}
    requirements: list[StaticRequirement] = []
    for assertion in (*contract.targets, *contract.guards):
        for record in _records_for(assertion, records):
            subject = _prompt_subject(record, assertion)
            if subject is None:
                continue
            requirements.extend(_numeric_requirements(contract, assertion, subject))
            if assertion.type in {
                AssertionType.EXACT_MATCH,
                AssertionType.NORMALIZED_EXACT_MATCH,
                AssertionType.FREE_GENERATION_MATCH,
            }:
                expected = _string_option_or_record(assertion, record, "expected")
                match_type = _string_option_or_record(assertion, record, "match_type")
                if expected is not None and match_type not in {"regex", "contains"}:
                    requirements.append(
                        StaticRequirement(
                            contract_id=contract.id,
                            subject=subject,
                            assertion_type="exact_output",
                            relation="exact_output",
                            value=expected,
                        )
                    )
            elif assertion.type is AssertionType.MULTIPLE_CHOICE_MARGIN:
                correct = _string_option_or_record(assertion, record, "correct_choice")
                raw_choices = record.get("choices", assertion.option("choices"))
                if (
                    correct is not None
                    and isinstance(raw_choices, list | tuple)
                    and all(isinstance(choice, str) for choice in raw_choices)
                ):
                    choices = cast(Sequence[str], raw_choices)
                    choice_set_hash = sha256_bytes("\x1f".join(sorted(choices)).encode("utf-8"))
                    requirements.append(
                        StaticRequirement(
                            contract_id=contract.id,
                            subject=f"{subject}:{choice_set_hash}",
                            assertion_type="multiple_choice_winner",
                            relation="equals",
                            value=correct,
                        )
                    )
            elif assertion.type is AssertionType.SEQUENCE_MARGIN:
                preferred = _string_option_or_record(assertion, record, "preferred")
                dispreferred = _string_option_or_record(assertion, record, "dispreferred")
                if preferred is not None and dispreferred is not None and preferred != dispreferred:
                    pair_hash = sha256_bytes(
                        "\x1f".join(sorted((preferred, dispreferred))).encode("utf-8")
                    )
                    requirements.append(
                        StaticRequirement(
                            contract_id=contract.id,
                            subject=f"{subject}:{pair_hash}",
                            assertion_type="sequence_preference",
                            relation="equals",
                            value=preferred,
                        )
                    )
    return tuple(requirements)


def check_static_contracts(
    contracts: Sequence[BehaviorContract],
    *,
    records_by_contract: Mapping[str, Mapping[str, Sequence[ProbeRecord]]] | None = None,
) -> StaticCheckResult:
    if not contracts:
        raise ValueError("at least one contract is required")
    identifiers = [contract.id for contract in contracts]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("contract identifiers must be unique for static checking")
    requirements: list[StaticRequirement] = []
    for contract in contracts:
        source_records = (records_by_contract or {}).get(contract.id, {})
        requirements.extend(
            contract_static_requirements(contract, records_by_source=source_records)
        )
    bases = {
        contract.id: contract.model_requirements.base_signature
        for contract in contracts
        if contract.model_requirements.base_signature is not None
    }
    tokenizers = {
        contract.id: contract.model_requirements.tokenizer_hash
        for contract in contracts
        if contract.model_requirements.tokenizer_hash is not None
    }
    identity_witnesses = incompatible_identity_requirements(
        base_signatures=bases,
        tokenizer_hashes=tokenizers,
    )
    witnesses = tuple(
        sorted(
            (*find_static_contradictions(requirements), *identity_witnesses),
            key=lambda item: (item.code, item.subject, item.contract_ids),
        )
    )
    return StaticCheckResult(
        status=(
            StaticCheckStatus.CONTRADICTION
            if witnesses
            else StaticCheckStatus.NO_CONTRADICTION_FOUND
        ),
        witnesses=witnesses,
        checked_contracts=tuple(sorted(identifiers)),
        requirements_examined=len(requirements),
    )


__all__ = [
    "ProbeRecord",
    "StaticCheckResult",
    "StaticCheckStatus",
    "check_static_contracts",
    "contract_static_requirements",
]
