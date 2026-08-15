"""Deterministic bounded-size shard planning."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor


def tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def plan_shards(
    tensors: Mapping[str, Tensor], *, max_shard_size: int
) -> tuple[tuple[str, ...], ...]:
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in sorted(tensors):
        size = tensor_nbytes(tensors[key])
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)
    return tuple(tuple(shard) for shard in shards)
