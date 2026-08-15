from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from modelpact.patch.ast import DeltaProgram, LowRankMatrixDelta, SparseMatrixDelta


@settings(deadline=None, max_examples=20)
@given(
    output_size=st.integers(min_value=1, max_value=8),
    input_size=st.integers(min_value=1, max_value=8),
    rank=st.integers(min_value=1, max_value=4),
)
def test_low_rank_materialization_is_additive_and_roundtrips(
    output_size: int, input_size: int, rank: int
) -> None:
    generator = torch.Generator().manual_seed(output_size * 100 + input_size * 10 + rank)
    base = torch.randn(output_size, input_size, generator=generator)
    left = torch.randn(output_size, rank, generator=generator)
    right = torch.randn(rank, input_size, generator=generator)
    tensors = {"left": left, "right": right}
    program = DeltaProgram({"weight": LowRankMatrixDelta("left", "right", 0.25)})
    parsed = DeltaProgram.from_dict(program.to_dict())
    result = parsed.apply_to_state({"weight": base}, tensors)["weight"]
    assert torch.allclose(result, base + 0.25 * (left @ right), rtol=0, atol=0)


@settings(deadline=None, max_examples=20)
@given(rows=st.integers(1, 8), columns=st.integers(1, 8))
def test_sparse_delta_matches_dense_indexing(rows: int, columns: int) -> None:
    count = min(rows * columns, 5)
    flat = torch.arange(count, dtype=torch.int64)
    indices = torch.stack((flat // columns, flat % columns), dim=1)
    values = torch.arange(1, count + 1, dtype=torch.float32)
    operation = SparseMatrixDelta("indices", "values", (rows, columns))
    result = operation.materialize({"indices": indices, "values": values})
    expected = torch.zeros(rows, columns)
    expected[indices[:, 0], indices[:, 1]] = values
    assert torch.equal(result, expected)
