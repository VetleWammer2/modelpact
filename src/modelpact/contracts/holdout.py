"""One-way access gate for sealed holdout data.

The gate is an execution-policy boundary, not cryptographic DRM.  Compiler code
must receive a gate instead of raw holdout paths; trusted orchestration selects
one final candidate and consumes the gate exactly once.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from modelpact.contracts.ast import BehaviorContract, UnsealPolicy
from modelpact.util.paths import resolve_inside


class HoldoutAccessError(PermissionError):
    pass


class HoldoutPhase(StrEnum):
    COMPILATION = "compilation"
    COUNTEREXAMPLE_SEARCH = "counterexample_search"
    VALIDATION = "validation"
    FINAL_CANDIDATE = "final_candidate"
    INDEPENDENT_VERIFICATION = "independent_verification"


class HoldoutRole(StrEnum):
    TARGETS = "targets"
    GUARDS = "guards"


@dataclass(frozen=True, slots=True)
class HoldoutCapability:
    contract_hash: str
    candidate_id: str
    phase: HoldoutPhase
    nonce: str


@dataclass(frozen=True, slots=True)
class HoldoutAccessRecord:
    sequence: int
    contract_hash: str
    candidate_id: str
    phase: HoldoutPhase
    role: HoldoutRole
    source: str


class SealedHoldoutGate:
    """Authorize a selected candidate without leaking results into optimization."""

    def __init__(self, contract: BehaviorContract, *, allow_independent: bool = False) -> None:
        self._contract = contract
        self._contract_hash = contract.contract_id
        self._allow_independent = allow_independent
        self._selected_candidate: str | None = None
        self._capability: HoldoutCapability | None = None
        self._consumed = False
        self._records: list[HoldoutAccessRecord] = []
        self._lock = threading.RLock()

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    @property
    def access_records(self) -> tuple[HoldoutAccessRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def select_final_candidate(self, candidate_id: str) -> None:
        if not candidate_id or len(candidate_id) > 4096 or "\x00" in candidate_id:
            raise ValueError("candidate_id must be a non-empty bounded string")
        with self._lock:
            if self._consumed or self._capability is not None:
                raise HoldoutAccessError("holdout has already been unsealed")
            if self._selected_candidate is not None and self._selected_candidate != candidate_id:
                raise HoldoutAccessError("a different final candidate was already selected")
            self._selected_candidate = candidate_id

    def authorize(
        self,
        *,
        phase: HoldoutPhase,
        candidate_id: str,
    ) -> HoldoutCapability:
        with self._lock:
            if not self._contract.holdout.configured:
                raise HoldoutAccessError("contract has no configured holdout")
            if phase in {
                HoldoutPhase.COMPILATION,
                HoldoutPhase.COUNTEREXAMPLE_SEARCH,
                HoldoutPhase.VALIDATION,
            }:
                raise HoldoutAccessError(f"sealed holdout is inaccessible during {phase.value}")
            if self._consumed:
                raise HoldoutAccessError("holdout authorization has already been consumed")
            if phase is HoldoutPhase.FINAL_CANDIDATE:
                if self._contract.holdout.unseal_policy is not UnsealPolicy.FINAL_CANDIDATE_ONLY:
                    message = "contract policy does not authorize final-candidate access"
                    raise HoldoutAccessError(message)
                if self._selected_candidate != candidate_id:
                    raise HoldoutAccessError("candidate was not selected as the final candidate")
            elif phase is HoldoutPhase.INDEPENDENT_VERIFICATION:
                if not self._allow_independent:
                    message = "this gate was not created for independent verification"
                    raise HoldoutAccessError(message)
            else:  # pragma: no cover - exhaustive enum guard for future versions
                raise HoldoutAccessError(f"unsupported holdout phase {phase!r}")
            capability = HoldoutCapability(
                contract_hash=self._contract_hash,
                candidate_id=candidate_id,
                phase=phase,
                nonce=secrets.token_hex(32),
            )
            self._capability = capability
            self._consumed = True
            return capability

    def validate(self, capability: HoldoutCapability, role: HoldoutRole) -> str:
        with self._lock:
            if self._capability is None or not secrets.compare_digest(
                capability.nonce, self._capability.nonce
            ):
                raise HoldoutAccessError("invalid holdout capability")
            if capability != self._capability or capability.contract_hash != self._contract_hash:
                raise HoldoutAccessError("holdout capability does not match this contract")
            source = (
                self._contract.holdout.targets
                if role is HoldoutRole.TARGETS
                else self._contract.holdout.guards
            )
            if source is None:
                raise HoldoutAccessError(f"contract has no {role.value} holdout source")
            record = HoldoutAccessRecord(
                sequence=len(self._records),
                contract_hash=self._contract_hash,
                candidate_id=capability.candidate_id,
                phase=capability.phase,
                role=role,
                source=source,
            )
            self._records.append(record)
            return source

    def read_bytes(
        self,
        capability: HoldoutCapability,
        *,
        role: HoldoutRole,
        contract_root: str | Path,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> bytes:
        """Read a holdout source after authorization and path validation."""

        if isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        source = self.validate(capability, role)
        path = resolve_inside(contract_root, source)
        size = path.stat().st_size
        if size > max_bytes:
            raise HoldoutAccessError(f"holdout source exceeds {max_bytes} bytes")
        return path.read_bytes()


__all__ = [
    "HoldoutAccessError",
    "HoldoutAccessRecord",
    "HoldoutCapability",
    "HoldoutPhase",
    "HoldoutRole",
    "SealedHoldoutGate",
]
