"""Checkpoint-level alias consistency checks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from modelpact.models.schema import ModelStateSchema


def validate_checkpoint_aliases(
    tensors: Mapping[str, Tensor], state_schema: ModelStateSchema
) -> None:
    for group in state_schema.aliases:
        present = [member for member in group.members if member in tensors]
        if present and len(present) != len(group.members):
            raise ValueError(f"checkpoint contains only part of alias group: {group.members}")
        if present:
            canonical = tensors[group.canonical]
            for member in group.members[1:]:
                if canonical.shape != tensors[member].shape or not torch.equal(
                    canonical, tensors[member]
                ):
                    raise ValueError(f"checkpoint alias values disagree: {group.members}")


def expand_checkpoint_aliases(
    tensors: Mapping[str, Tensor], state_schema: ModelStateSchema
) -> dict[str, Tensor]:
    """Expand physically omitted tied keys after checking every stored copy.

    SafeTensors writers commonly store one member of a tied embedding/output
    group. Materialization needs a logical tensor for every declared target; the
    resulting mapping uses clones so a writer never encounters shared storage.
    """

    result = dict(tensors)
    for group in state_schema.aliases:
        present = [member for member in group.members if member in result]
        if not present:
            continue
        reference = result[present[0]]
        for member in present[1:]:
            if reference.shape != result[member].shape or not torch.equal(
                reference, result[member]
            ):
                raise ValueError(f"checkpoint alias values disagree: {group.members}")
        for member in group.members:
            result.setdefault(member, reference.clone())
    return dict(sorted(result.items()))
