"""Compiler evidence and honest terminal states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor


class CompilationStatus(StrEnum):
    FEASIBLE = "FEASIBLE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    HOLDOUT_FAILED = "HOLDOUT_FAILED"
    INFEASIBLE_WITHIN_BUDGET = "INFEASIBLE_WITHIN_BUDGET"
    RESOURCE_BUDGET_EXHAUSTED = "RESOURCE_BUDGET_EXHAUSTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class StepEvidence:
    step: int
    target_losses: dict[str, float]
    guard_margins: dict[str, float]
    multipliers: dict[str, float]
    gradient_norm: float
    patch_norm: float
    feasible: bool


@dataclass(slots=True)
class CompilationResult:
    status: CompilationStatus
    deltas: dict[str, Tensor]
    factors: dict[str, tuple[Tensor, Tensor]]
    active_modules: tuple[str, ...]
    ranks: dict[str, int]
    evidence: list[StepEvidence] = field(default_factory=list)
    best_step: int | None = None
    best_target_loss: float | None = None
    violated_constraints: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.status is CompilationStatus.FEASIBLE

    def detached_cpu(self) -> CompilationResult:
        return CompilationResult(
            status=self.status,
            deltas={name: value.detach().cpu().clone() for name, value in self.deltas.items()},
            factors={
                name: (left.detach().cpu().clone(), right.detach().cpu().clone())
                for name, (left, right) in self.factors.items()
            },
            active_modules=self.active_modules,
            ranks=dict(self.ranks),
            evidence=list(self.evidence),
            best_step=self.best_step,
            best_target_loss=self.best_target_loss,
            violated_constraints=dict(self.violated_constraints),
            warnings=list(self.warnings),
            metadata=dict(self.metadata),
        )


def patch_frobenius_norm(factors: dict[str, tuple[Tensor, Tensor]]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for left, right in factors.values():
        delta = left.to(torch.float64) @ right.to(torch.float64)
        squared += delta.square().sum()
    return float(squared.sqrt().item())
