"""Bounded projected activation difference fingerprints."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def projected_difference(base: Tensor, target: Tensor, *, dimensions: int = 16, seed: int = 0) -> tuple[float, ...]:
    if base.shape != target.shape:
        raise ValueError("activation shapes differ")
    flattened = (target.detach().to(torch.float64) - base.detach().to(torch.float64)).reshape(-1)
    if flattened.numel() == 0:
        return ()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn((dimensions, flattened.numel()), generator=generator, dtype=torch.float64)
    projection /= max(1.0, flattened.numel() ** 0.5)
    return tuple(float(item) for item in (projection @ flattened.cpu()).tolist())


def concatenate_fingerprints(values: Iterable[tuple[float, ...]], *, maximum_values: int = 256) -> tuple[float, ...]:
    output: list[float] = []
    for value in values:
        output.extend(value)
        if len(output) >= maximum_values:
            break
    return tuple(output[:maximum_values])

