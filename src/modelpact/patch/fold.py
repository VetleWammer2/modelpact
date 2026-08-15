"""Fold delta programs into state mappings or new SafeTensors checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

from modelpact.checkpoints.writer import DEFAULT_MAX_SHARD_SIZE, materialize_checkpoint
from modelpact.models.manifest import ModelSignature
from modelpact.models.schema import ModelStateSchema
from modelpact.patch.ast import DeltaProgram
from modelpact.patch.bundle import PatchBundle
from modelpact.patch.validate import validate_base_signature


def fold_state_dict(
    state: Mapping[str, Tensor],
    program: DeltaProgram,
    tensors: Mapping[str, Tensor],
    *,
    state_schema: ModelStateSchema | None = None,
) -> dict[str, Tensor]:
    return program.apply_to_state(state, tensors, state_schema=state_schema)


def materialize_patch(
    source_checkpoint: str | Path,
    output: str | Path,
    program: DeltaProgram,
    tensors: Mapping[str, Tensor],
    *,
    state_schema: ModelStateSchema | None = None,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    patch_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return materialize_checkpoint(
        source_checkpoint,
        output,
        program,
        tensors,
        state_schema=state_schema,
        max_shard_size=max_shard_size,
        patch_ids=patch_ids,
    )


def materialize_bundle(
    source_checkpoint: str | Path,
    output: str | Path,
    bundle: PatchBundle,
    actual_signature: ModelSignature | Mapping[str, object],
    *,
    state_schema: ModelStateSchema | None = None,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> dict[str, object]:
    """Validate exact base identity before folding a bundle into a checkpoint."""

    validate_base_signature(bundle.manifest.base_signature, actual_signature)
    return materialize_patch(
        source_checkpoint,
        output,
        bundle.program,
        bundle.tensors,
        state_schema=state_schema,
        max_shard_size=max_shard_size,
        patch_ids=(bundle.manifest.patch_id,),
    )
