"""Numerically stable behavioral divergence metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def _probabilities(logits: Tensor) -> Tensor:
    if logits.ndim < 1:
        raise ValueError("logits require at least one dimension")
    return torch.softmax(logits.to(torch.float64), dim=-1)


def symmetric_kl(base_logits: Tensor, target_logits: Tensor) -> Tensor:
    if base_logits.shape != target_logits.shape:
        raise ValueError("logit shapes differ")
    p = _probabilities(base_logits).clamp_min(1e-12)
    q = _probabilities(target_logits).clamp_min(1e-12)
    return 0.5 * ((p * (p.log() - q.log())).sum(dim=-1) + (q * (q.log() - p.log())).sum(dim=-1))


def jensen_shannon(base_logits: Tensor, target_logits: Tensor) -> Tensor:
    if base_logits.shape != target_logits.shape:
        raise ValueError("logit shapes differ")
    p = _probabilities(base_logits).clamp_min(1e-12)
    q = _probabilities(target_logits).clamp_min(1e-12)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1) + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )


def top_token_flip_rate(base_logits: Tensor, target_logits: Tensor) -> float:
    if base_logits.shape != target_logits.shape:
        raise ValueError("logit shapes differ")
    return float(
        (base_logits.argmax(dim=-1) != target_logits.argmax(dim=-1)).to(torch.float64).mean().item()
    )


def output_interaction_residual(
    base: Tensor, left: Tensor, right: Tensor, combined: Tensor
) -> Tensor:
    for tensor in (left, right, combined):
        if tensor.shape != base.shape:
            raise ValueError("output shapes differ")
    return combined - left - right + base
