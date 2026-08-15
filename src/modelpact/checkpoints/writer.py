"""Non-overwriting deterministic SafeTensors checkpoint materialization."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

from modelpact.checkpoints.aliases import expand_checkpoint_aliases
from modelpact.checkpoints.safetensors import save_safetensors_atomic, tensor_content_hash
from modelpact.checkpoints.sharded import plan_shards, tensor_nbytes
from modelpact.checkpoints.store import checkpoint_files, load_checkpoint_tensors
from modelpact.models.schema import ModelStateSchema
from modelpact.patch.ast import DeltaProgram
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_file

DEFAULT_MAX_SHARD_SIZE = 2 * 1024**3
MAX_AUXILIARY_FILE_BYTES = 1024**3


def _source_file_hashes(source: Path) -> dict[str, str]:
    if source.is_file():
        return {source.name: sha256_file(source)}
    result = {}
    for path in sorted(
        item for item in source.iterdir() if item.is_file() and not item.is_symlink()
    ):
        result[path.name] = sha256_file(path)
    return result


def _copy_auxiliary_files(source: Path, temporary: Path) -> tuple[str, ...]:
    if source.is_file():
        return ()
    copied = []
    excluded = {"model.safetensors.index.json", "materialization-manifest.json"}
    for path in sorted(source.iterdir()):
        if path.name in excluded or path.suffix == ".safetensors":
            continue
        if path.is_symlink():
            raise ValueError(f"checkpoint auxiliary file may not be a symlink: {path.name}")
        if path.is_dir():
            # Arbitrary recursive copying expands the trust surface. Tokenizer
            # assets used by supported adapters are regular top-level files.
            continue
        if path.stat().st_size > MAX_AUXILIARY_FILE_BYTES:
            raise ValueError(f"checkpoint auxiliary file exceeds size limit: {path.name}")
        shutil.copyfile(path, temporary / path.name)
        copied.append(path.name)
    return tuple(copied)


def materialize_checkpoint(
    source_checkpoint: str | Path,
    output: str | Path,
    program: DeltaProgram,
    patch_tensors: Mapping[str, Tensor],
    *,
    state_schema: ModelStateSchema | None = None,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    patch_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Fold an additive program into a new checkpoint through an atomic directory rename."""

    source = Path(source_checkpoint)
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    # Resolve parents without requiring the not-yet-created output path.
    source_resolved = source.resolve()
    target_resolved = target.parent.resolve() / target.name
    if source_resolved == target_resolved or source_resolved in target_resolved.parents:
        raise ValueError("output must not be the source checkpoint or nested within it")
    checkpoint_files(source)  # validate the source before allocating output state
    before_hashes = _source_file_hashes(source)
    base_tensors = load_checkpoint_tensors(source)
    if state_schema is not None:
        base_tensors = expand_checkpoint_aliases(base_tensors, state_schema)
    patched = program.apply_to_state(base_tensors, patch_tensors, state_schema=state_schema)
    shards = plan_shards(patched, max_shard_size=max_shard_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    output_files: list[str] = []
    weight_map: dict[str, str] = {}
    try:
        auxiliary_files = _copy_auxiliary_files(source, temporary)
        shard_count = len(shards)
        for index, keys in enumerate(shards, start=1):
            filename = (
                "model.safetensors"
                if shard_count == 1
                else f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            )
            save_safetensors_atomic(
                temporary / filename,
                {key: patched[key] for key in keys},
                metadata={"format": "modelpact-materialized-v1"},
                overwrite=False,
            )
            output_files.append(filename)
            weight_map.update(dict.fromkeys(keys, filename))
        if shard_count > 1:
            index_value = {
                "metadata": {"total_size": sum(tensor_nbytes(value) for value in patched.values())},
                "weight_map": dict(sorted(weight_map.items())),
            }
            atomic_write_text(
                temporary / "model.safetensors.index.json",
                canonical_dumps(index_value),
                overwrite=False,
            )
            output_files.append("model.safetensors.index.json")
        manifest: dict[str, object] = {
            "auxiliary_files": list(auxiliary_files),
            "output_files": sorted(output_files),
            "output_tensor_hashes": {
                key: tensor_content_hash(value) for key, value in sorted(patched.items())
            },
            "patch_ids": list(patch_ids),
            "resolved_delta_program_hash": hash_canonical(program.to_dict()),
            "schema_version": 1,
            "source_file_hashes": before_hashes,
            "source_path_record": source.name,
        }
        atomic_write_text(
            temporary / "materialization-manifest.json",
            canonical_dumps(manifest),
            overwrite=False,
        )
        after_hashes = _source_file_hashes(source)
        if after_hashes != before_hashes:
            raise RuntimeError("source checkpoint changed during materialization")
        os.replace(temporary, target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
