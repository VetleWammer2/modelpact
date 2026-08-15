"""Deterministic parameter-space merging baselines.

These functions only transform deltas. They do not verify behavior and must not
be presented as semantic merges without an external executed contract result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

DeltaState = Mapping[str, Tensor]


def _validate(states: Sequence[DeltaState]) -> tuple[str, ...]:
    if not states:
        raise ValueError("at least one delta state is required")
    keys = tuple(sorted(states[0]))
    for state in states:
        if tuple(sorted(state)) != keys:
            raise ValueError("delta states have different keys")
        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(f"delta shape mismatch for {key}")
            if state[key].dtype != states[0][key].dtype:
                raise ValueError(f"delta dtype mismatch for {key}")
    return keys


def weighted_delta_sum(
    states: Sequence[DeltaState], weights: Sequence[float] | None = None
) -> dict[str, Tensor]:
    keys = _validate(states)
    coefficients = tuple(weights) if weights is not None else (1.0,) * len(states)
    if len(coefficients) != len(states) or not all(
        torch.isfinite(torch.tensor(value)) for value in coefficients
    ):
        raise ValueError("weights must be finite and match the state count")
    return {
        key: sum(
            (
                state[key] * coefficient
                for state, coefficient in zip(states, coefficients, strict=True)
            ),
            torch.zeros_like(states[0][key]),
        )
        for key in keys
    }


def task_arithmetic(states: Sequence[DeltaState], *, scale: float = 1.0) -> dict[str, Tensor]:
    return weighted_delta_sum(states, [scale] * len(states))


def ties_merge(states: Sequence[DeltaState], *, density: float = 0.2) -> dict[str, Tensor]:
    """TRIM, ELECT SIGN, and merge matching-sign entries."""

    keys = _validate(states)
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    result: dict[str, Tensor] = {}
    for key in keys:
        stacked = torch.stack([state[key].to(torch.float64) for state in states])
        trimmed = torch.zeros_like(stacked)
        keep = max(1, round(stacked.shape[1:].numel() * density))
        for index, tensor in enumerate(stacked):
            flattened = tensor.abs().reshape(-1)
            threshold_indices = torch.topk(flattened, keep, sorted=False).indices
            mask = torch.zeros_like(flattened, dtype=torch.bool)
            mask[threshold_indices] = True
            trimmed[index] = torch.where(mask.reshape_as(tensor), tensor, torch.zeros_like(tensor))
        elected = torch.sign(trimmed.sum(dim=0))
        aligned = torch.sign(trimmed) == elected.unsqueeze(0)
        count = aligned.sum(dim=0).clamp_min(1)
        merged = torch.where(aligned, trimmed, torch.zeros_like(trimmed)).sum(dim=0) / count
        result[key] = merged.to(states[0][key].dtype)
    return result


def dare(
    states: Sequence[DeltaState], *, drop_probability: float = 0.5, seed: int = 0
) -> dict[str, Tensor]:
    """Drop and rescale each task delta before additive merging."""

    keys = _validate(states)
    if not 0 <= drop_probability < 1:
        raise ValueError("drop_probability must be in [0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    retained = 1.0 - drop_probability
    output: dict[str, Tensor] = {}
    for key in keys:
        merged = torch.zeros_like(states[0][key])
        for state in states:
            mask = torch.rand(state[key].shape, generator=generator, device="cpu") < retained
            merged = (
                merged
                + state[key] * mask.to(device=state[key].device, dtype=state[key].dtype) / retained
            )
        output[key] = merged
    return output


def cat_projection(states: Sequence[DeltaState]) -> dict[str, Tensor]:
    """Small CAT-style projection baseline for matrix deltas.

    Later deltas have components opposed to the running anchor projected away.
    This is documented as a local baseline, not a full reproduction of CAT.
    """

    keys = _validate(states)
    output: dict[str, Tensor] = {}
    for key in keys:
        anchor = states[0][key].to(torch.float64).clone()
        for state in states[1:]:
            candidate = state[key].to(torch.float64)
            inner = (anchor * candidate).sum()
            if inner < 0:
                candidate = candidate - inner / anchor.square().sum().clamp_min(1e-12) * anchor
            anchor = anchor + candidate
        output[key] = anchor.to(states[0][key].dtype)
    return output
