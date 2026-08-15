"""Conservative static contract-contradiction checks.

This module intentionally works on a small normalized requirement record instead
of depending on the contract parser.  The parser can lower assertions into these
records; callers with richer semantics can supply an additional checker to the
composition orchestration layer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, TypeAlias

Scalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class StaticRequirement:
    """A statically comparable fragment of a behavior contract.

    ``subject`` is normally a prompt hash plus the scored choice/sequence.  Only
    requirements sharing a subject and assertion type are compared.
    """

    contract_id: str
    subject: str
    assertion_type: str
    relation: str
    value: Scalar


@dataclass(frozen=True, slots=True)
class ContradictionWitness:
    code: str
    contract_ids: tuple[str, ...]
    subject: str
    explanation: str
    values: tuple[Scalar, ...]


class ContradictionChecker(Protocol):
    def __call__(
        self, contract_ids: tuple[str, ...]
    ) -> tuple[ContradictionWitness, ...]: ...


_EXACT_RELATIONS = frozenset({"equals", "exact_output"})
_LOWER_RELATIONS = frozenset({">", ">="})
_UPPER_RELATIONS = frozenset({"<", "<="})


def find_static_contradictions(
    requirements: tuple[StaticRequirement, ...] | list[StaticRequirement],
) -> tuple[ContradictionWitness, ...]:
    """Return only contradictions justified by normalized requirements.

    Absence of a result says nothing about joint satisfiability.
    """

    groups: dict[tuple[str, str], list[StaticRequirement]] = defaultdict(list)
    for requirement in requirements:
        groups[(requirement.subject, requirement.assertion_type)].append(requirement)

    witnesses: list[ContradictionWitness] = []
    for (subject, assertion_type), group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda item: (item.contract_id, item.relation, repr(item.value)),
        )
        exact = [item for item in ordered if item.relation in _EXACT_RELATIONS]
        distinct_exact: list[StaticRequirement] = []
        for item in exact:
            if not any(item.value == prior.value for prior in distinct_exact):
                distinct_exact.append(item)
        if len(distinct_exact) > 1:
            witnesses.append(
                ContradictionWitness(
                    code="INCOMPATIBLE_EXACT_REQUIREMENTS",
                    contract_ids=tuple(item.contract_id for item in distinct_exact),
                    subject=subject,
                    explanation=(
                        f"{assertion_type} requires different exact values for the same subject"
                    ),
                    values=tuple(item.value for item in distinct_exact),
                )
            )
            continue

        numeric = [
            item
            for item in ordered
            if item.relation in _LOWER_RELATIONS | _UPPER_RELATIONS
            and isinstance(item.value, int | float)
            and not isinstance(item.value, bool)
        ]
        lowers = [item for item in numeric if item.relation in _LOWER_RELATIONS]
        uppers = [item for item in numeric if item.relation in _UPPER_RELATIONS]
        if not lowers or not uppers:
            continue
        def numeric_value(requirement: StaticRequirement) -> float:
            value = requirement.value
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError("numeric interval requirement has a nonnumeric value")
            return float(value)

        strongest_lower = max(lowers, key=numeric_value)
        strongest_upper = min(uppers, key=numeric_value)
        lower_value = numeric_value(strongest_lower)
        upper_value = numeric_value(strongest_upper)
        equality_allowed = (
            lower_value == upper_value
            and strongest_lower.relation == ">="
            and strongest_upper.relation == "<="
        )
        if lower_value > upper_value or (lower_value == upper_value and not equality_allowed):
            witnesses.append(
                ContradictionWitness(
                    code="EMPTY_NUMERIC_INTERVAL",
                    contract_ids=(strongest_lower.contract_id, strongest_upper.contract_id),
                    subject=subject,
                    explanation=f"{assertion_type} declares an empty numeric acceptance interval",
                    values=(strongest_lower.value, strongest_upper.value),
                )
            )
    return tuple(witnesses)


def incompatible_identity_requirements(
    *,
    base_signatures: dict[str, str],
    tokenizer_hashes: dict[str, str],
) -> tuple[ContradictionWitness, ...]:
    """Find exact base/tokenizer requirements that cannot share one execution."""

    witnesses: list[ContradictionWitness] = []
    for code, subject, values in (
        ("INCOMPATIBLE_BASE_SIGNATURES", "base_signature", base_signatures),
        ("INCOMPATIBLE_TOKENIZERS", "tokenizer_hash", tokenizer_hashes),
    ):
        distinct = sorted(set(values.values()))
        if len(distinct) > 1:
            witnesses.append(
                ContradictionWitness(
                    code=code,
                    contract_ids=tuple(sorted(values)),
                    subject=subject,
                    explanation=f"contracts require {len(distinct)} incompatible {subject} values",
                    values=tuple(distinct),
                )
            )
    return tuple(witnesses)
