"""Augmented-Lagrangian state for explicit preservation constraints."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


LossFunction = Callable[[nn.Module, Any], Tensor]


@dataclass(frozen=True, slots=True)
class DifferentiableObjective:
    objective_id: str
    batches: tuple[Any, ...]
    loss: LossFunction
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.objective_id or not self.batches:
            raise ValueError("objective requires an id and at least one batch")
        if self.weight <= 0:
            raise ValueError("objective weight must be positive")


@dataclass(frozen=True, slots=True)
class DifferentiableConstraint:
    constraint_id: str
    batches: tuple[Any, ...]
    measure: LossFunction
    maximum: float

    def __post_init__(self) -> None:
        if not self.constraint_id or not self.batches:
            raise ValueError("constraint requires an id and at least one batch")
        if not torch.isfinite(torch.tensor(self.maximum)):
            raise ValueError("constraint maximum must be finite")


@dataclass(slots=True)
class MultiplierState:
    value: float = 0.0
    learning_rate: float = 0.1
    maximum: float = 1_000_000.0

    def update(self, violation: float) -> None:
        self.value = min(self.maximum, max(0.0, self.value + self.learning_rate * violation))


def mean_loss(model: nn.Module, batches: Iterable[Any], function: LossFunction) -> Tensor:
    losses = [function(model, batch) for batch in batches]
    if not losses:
        raise ValueError("cannot evaluate an empty batch collection")
    if any(loss.ndim != 0 for loss in losses):
        raise ValueError("objective and constraint functions must return scalar tensors")
    return torch.stack(losses).mean()


def augmented_penalty(violation: Tensor, multiplier: float, rho: float) -> Tensor:
    positive = torch.relu(violation)
    return multiplier * positive + rho * positive.square()

