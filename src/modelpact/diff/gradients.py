"""Projected gradient fingerprints without retaining per-example gradients."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def gradient_fingerprint(
    gradients: Iterable[Tensor | None], *, dimensions: int = 16, seed: int = 0
) -> tuple[float, ...]:
    flattened = [gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1) for gradient in gradients if gradient is not None]
    if not flattened:
        return ()
    vector = torch.cat(flattened)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn((dimensions, vector.numel()), generator=generator, dtype=torch.float64)
    projection /= max(1.0, vector.numel() ** 0.5)
    return tuple(float(item) for item in (projection @ vector).tolist())

