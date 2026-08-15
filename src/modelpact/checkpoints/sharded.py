"""Deterministic bounded-size shard planning."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor


def tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def plan_shards(
    tensors: Mapping[str, Tensor], *, max_shard_size: int
) -> tuple[tuple[str, ...], ...]:
    return plan_shards_by_size(
        {key: tensor_nbytes(value) for key, value in tensors.items()},
        max_shard_size=max_shard_size,
    )


def plan_shards_by_size(
    tensor_sizes: Mapping[str, int], *, max_shard_size: int
) -> tuple[tuple[str, ...], ...]:
    """Plan deterministic shards without materializing the tensors themselves."""

    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in sorted(tensor_sizes):
        size = tensor_sizes[key]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid tensor size for {key}")
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)
    return tuple(tuple(shard) for shard in shards)
