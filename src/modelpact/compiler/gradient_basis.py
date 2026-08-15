"""Bounded deterministic low-rank bases from contrastive gradients."""

from __future__ import annotations

import torch
from torch import Tensor


def randomized_svd(
    matrix: Tensor,
    *,
    rank: int,
    oversampling: int = 4,
    power_iterations: int = 1,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    if matrix.ndim != 2:
        raise ValueError("randomized_svd expects a matrix")
    maximum_rank = min(matrix.shape)
    if rank <= 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [1, {maximum_rank}]")
    width = min(maximum_rank, rank + max(0, oversampling))
    working = matrix.to(torch.float64)
    generator = torch.Generator(device=working.device).manual_seed(seed)
    omega = torch.randn((working.shape[1], width), generator=generator, device=working.device, dtype=working.dtype)
    sample = working @ omega
    for _ in range(power_iterations):
        sample = working @ (working.mT @ sample)
    q, _ = torch.linalg.qr(sample, mode="reduced")
    reduced = q.mT @ working
    u_hat, singular, vh = torch.linalg.svd(reduced, full_matrices=False)
    u = q @ u_hat
    return (
        u[:, :rank].to(matrix.dtype),
        singular[:rank].to(matrix.dtype),
        vh[:rank, :].to(matrix.dtype),
    )


def direct_or_randomized_svd(matrix: Tensor, *, rank: int, direct_limit: int = 1_000_000, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    if matrix.numel() <= direct_limit:
        u, singular, vh = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
        return u[:, :rank].to(matrix.dtype), singular[:rank].to(matrix.dtype), vh[:rank].to(matrix.dtype)
    return randomized_svd(matrix, rank=rank, seed=seed)


def low_rank_factors(matrix: Tensor, *, rank: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    u, singular, vh = direct_or_randomized_svd(matrix, rank=rank, seed=seed)
    root = singular.clamp_min(0).sqrt()
    return u * root.unsqueeze(0), root.unsqueeze(1) * vh

