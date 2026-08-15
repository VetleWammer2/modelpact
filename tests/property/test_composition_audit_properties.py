from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from modelpact.audit.reduce import ddmin_failing_subset
from modelpact.audit.subsets import subset_to_vector, vector_to_subset
from modelpact.compose.closure import PatchOperand, additive_compose


@settings(deadline=None, max_examples=50)
@given(
    st.lists(
        st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_additive_patch_resolution_is_permutation_invariant(values: list[float]) -> None:
    patches = [
        PatchOperand(
            patch_id=f"p{index:02d}",
            base_signature="base",
            module_schema_hash="schema",
            delta={"weight": torch.tensor([value], dtype=torch.float64)},
            contract_ids=(f"c{index:02d}",),
        )
        for index, value in enumerate(values)
    ]
    forward = additive_compose(patches)["weight"]
    reverse = additive_compose(list(reversed(patches)))["weight"]
    assert torch.equal(forward, reverse)


@given(
    st.sets(st.sampled_from(("a", "b", "c", "d", "e"))),
)
@settings(deadline=None, max_examples=50)
def test_subset_vector_round_trip(members: set[str]) -> None:
    universe = ("e", "d", "c", "b", "a")
    vector = subset_to_vector(members, universe)
    assert vector_to_subset(vector, universe) == tuple(sorted(members))


@given(
    required=st.sets(st.sampled_from(("a", "b", "c", "d")), min_size=1),
    extras=st.sets(st.sampled_from(("a", "b", "c", "d"))),
)
@settings(deadline=None, max_examples=50)
def test_ddmin_result_is_one_minimal_for_monotone_failures(
    required: set[str], extras: set[str]
) -> None:
    initial = tuple(sorted(required | extras))
    result = ddmin_failing_subset(
        initial,
        oracle=lambda subset: required <= set(subset),
    )
    assert required <= set(result.reduced)
    assert all(not required <= (set(result.reduced) - {member}) for member in result.reduced)
