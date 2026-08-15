"""Serializable evidence records for behavior rebases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from modelpact.status import RebaseClaim


@dataclass(frozen=True, slots=True)
class RebaseEvidence:
    source_patch_id: str
    source_base_hash: str
    target_base_hash: str
    claim: RebaseClaim
    compatibility: str
    direct_attempted: bool
    direct_outcome: str | None
    recompile_attempted: bool
    recompile_steps: int
    recompile_restarts: int
    budget_exhausted: bool
    old_patched_behavior: Mapping[str, float] = field(default_factory=dict)
    new_patched_behavior: Mapping[str, float] = field(default_factory=dict)
    new_base_preservation: Mapping[str, float] = field(default_factory=dict)
    patch_complexity_before: Mapping[str, int | float] = field(default_factory=dict)
    patch_complexity_after: Mapping[str, int | float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_patch_id": self.source_patch_id,
            "source_base_hash": self.source_base_hash,
            "target_base_hash": self.target_base_hash,
            "claim": self.claim.value,
            "compatibility": self.compatibility,
            "direct_attempted": self.direct_attempted,
            "direct_outcome": self.direct_outcome,
            "recompile_attempted": self.recompile_attempted,
            "recompile_steps": self.recompile_steps,
            "recompile_restarts": self.recompile_restarts,
            "budget_exhausted": self.budget_exhausted,
            "old_patched_behavior": dict(sorted(self.old_patched_behavior.items())),
            "new_patched_behavior": dict(sorted(self.new_patched_behavior.items())),
            "new_base_preservation": dict(sorted(self.new_base_preservation.items())),
            "patch_complexity_before": dict(sorted(self.patch_complexity_before.items())),
            "patch_complexity_after": dict(sorted(self.patch_complexity_after.items())),
            "warnings": list(self.warnings),
        }
