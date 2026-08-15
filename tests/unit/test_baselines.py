from __future__ import annotations

import torch

from modelpact.baselines.merging import dare, ties_merge, weighted_delta_sum


def test_weighted_sum_is_exact_and_deterministic() -> None:
    states = ({"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([3.0, 4.0])})
    torch.testing.assert_close(weighted_delta_sum(states)["w"], torch.tensor([4.0, 6.0]))


def test_ties_rejects_sign_conflict() -> None:
    states = ({"w": torch.tensor([3.0, 1.0])}, {"w": torch.tensor([-1.0, 1.0])})
    merged = ties_merge(states, density=1.0)["w"]
    assert merged[0] > 0
    assert merged[1] == 1


def test_dare_seed_is_reproducible() -> None:
    states = ({"w": torch.arange(20.0)}, {"w": torch.arange(20.0)})
    torch.testing.assert_close(dare(states, seed=7)["w"], dare(states, seed=7)["w"])
