"""Non-overwriting deterministic SafeTensors checkpoint materialization."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from torch import Tensor

from modelpact.checkpoints.safetensors import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TENSOR_ELEMENTS,
    DEFAULT_MAX_TENSORS,
    save_safetensors_atomic,
    tensor_content_hash,
)
from modelpact.checkpoints.sharded import plan_shards_by_size
from modelpact.checkpoints.store import MAX_INDEX_BYTES, checkpoint_files
from modelpact.models.schema import ModelStateSchema, dtype_name
from modelpact.patch.ast import DeltaProgram
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_file

DEFAULT_MAX_SHARD_SIZE = 2 * 1024**3
MAX_AUXILIARY_FILE_BYTES = 1024**3
MAX_SOURCE_FILES = 10_000
MAX_SOURCE_AGGREGATE_BYTES = 64 * 1024**3
_SAFETENSORS_DTYPE_INFO: dict[str, tuple[str, int]] = {
    "BF16": ("bfloat16", 2),
    "BOOL": ("bool", 1),
    "C128": ("complex128", 16),
    "C64": ("complex64", 8),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E4M3FN": ("float8_e4m3fn", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "I16": ("int16", 2),
    "I32": ("int32", 4),
    "I64": ("int64", 8),
    "I8": ("int8", 1),
    "U16": ("uint16", 2),
    "U32": ("uint32", 4),
    "U64": ("uint64", 8),
    "U8": ("uint8", 1),
}


@dataclass(frozen=True, slots=True)
class _StoredTensor:
    path: Path
    storage_key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(slots=True)
class _IOMeasurements:
    read_bytes: int = 0
    read_seconds: float = 0.0
    write_bytes: int = 0
    write_seconds: float = 0.0


def _source_file_hashes(source: Path) -> dict[str, str]:
    if source.is_file():
        return {source.name: sha256_file(source, max_bytes=DEFAULT_MAX_FILE_BYTES)}
    files: list[tuple[Path, int]] = []
    aggregate = 0
    for path in sorted(source.iterdir()):
        if path.is_symlink():
            raise ValueError(f"checkpoint source file may not be a symlink: {path.name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"checkpoint source entry must be a regular file: {path.name}")
        if path.suffix == ".safetensors":
            limit = DEFAULT_MAX_FILE_BYTES
        elif path.name == "model.safetensors.index.json":
            limit = MAX_INDEX_BYTES
        else:
            limit = MAX_AUXILIARY_FILE_BYTES
        size = path.stat().st_size
        if size > limit:
            raise ValueError(f"checkpoint source file exceeds size limit: {path.name}")
        aggregate += size
        if aggregate > MAX_SOURCE_AGGREGATE_BYTES:
            raise ValueError("checkpoint source files exceed the aggregate size limit")
        files.append((path, limit))
        if len(files) > MAX_SOURCE_FILES:
            raise ValueError("checkpoint source contains too many files")
    result = {}
    for path, limit in files:
        result[path.name] = sha256_file(path, max_bytes=limit)
    return result


def _checkpoint_tensor_index(files: tuple[Path, ...]) -> dict[str, _StoredTensor]:
    """Read only SafeTensors headers and build a bounded physical tensor index."""

    result: dict[str, _StoredTensor] = {}
    for shard in files:
        # SafeTensors currently ships no complete typing for safe_open or PySafeSlice.
        with safe_open(shard, framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
            keys = sorted(handle.keys())
            if len(keys) > DEFAULT_MAX_TENSORS:
                raise ValueError(f"SafeTensors key count exceeds {DEFAULT_MAX_TENSORS}: {shard}")
            for key in keys:
                if not key or len(key) > 2048 or "\x00" in key:
                    raise ValueError("invalid tensor key")
                if key in result:
                    raise ValueError(f"duplicate tensor key across checkpoint shards: {key}")
                view = handle.get_slice(key)
                shape = tuple(int(dimension) for dimension in view.get_shape())
                if len(shape) > 32 or any(
                    dimension < 0 or dimension > DEFAULT_MAX_TENSOR_ELEMENTS for dimension in shape
                ):
                    raise ValueError(f"invalid tensor shape in checkpoint: {key}")
                elements = prod(shape)
                if elements > DEFAULT_MAX_TENSOR_ELEMENTS:
                    raise ValueError(f"tensor exceeds element bound: {key}")
                safe_dtype = str(view.get_dtype())
                dtype_info = _SAFETENSORS_DTYPE_INFO.get(safe_dtype)
                if dtype_info is None:
                    raise ValueError(f"unsupported SafeTensors dtype {safe_dtype!r}: {key}")
                dtype, element_size = dtype_info
                result[key] = _StoredTensor(
                    path=shard,
                    storage_key=key,
                    shape=shape,
                    dtype=dtype,
                    nbytes=elements * element_size,
                )
                if len(result) > DEFAULT_MAX_TENSORS:
                    raise ValueError(f"checkpoint tensor count exceeds {DEFAULT_MAX_TENSORS}")
    return dict(sorted(result.items()))


def _load_stored_tensor(location: _StoredTensor, measurements: _IOMeasurements) -> Tensor:
    started = time.perf_counter()
    with safe_open(location.path, framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
        value = cast(Tensor, handle.get_tensor(location.storage_key)).clone()
    measurements.read_seconds += time.perf_counter() - started
    measurements.read_bytes += location.nbytes
    if (
        tuple(value.shape) != location.shape
        or dtype_name(value.dtype) != location.dtype
        or value.numel() * value.element_size() != location.nbytes
    ):
        raise RuntimeError(
            f"checkpoint tensor metadata changed while reading: {location.storage_key}"
        )
    return value


def _expand_alias_index(
    tensors: Mapping[str, _StoredTensor],
    state_schema: ModelStateSchema | None,
    measurements: _IOMeasurements,
) -> dict[str, _StoredTensor]:
    """Verify stored alias copies and add logical entries for omitted tied keys."""

    result = dict(tensors)
    if state_schema is None:
        return result
    for group in state_schema.aliases:
        present = [member for member in group.members if member in result]
        if not present:
            continue
        reference = result[present[0]]
        for member in present[1:]:
            candidate = result[member]
            if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
                raise ValueError(f"checkpoint alias values disagree: {group.members}")
        if len(present) > 1:
            reference_hash: str | None = None
            for member in present:
                value = _load_stored_tensor(result[member], measurements)
                candidate_hash = tensor_content_hash(value)
                del value
                if reference_hash is None:
                    reference_hash = candidate_hash
                elif candidate_hash != reference_hash:
                    raise ValueError(f"checkpoint alias values disagree: {group.members}")
        for member in group.members:
            result.setdefault(
                member,
                _StoredTensor(
                    path=reference.path,
                    storage_key=reference.storage_key,
                    shape=reference.shape,
                    dtype=reference.dtype,
                    nbytes=reference.nbytes,
                ),
            )
    return dict(sorted(result.items()))


def _validate_preflight(
    program: DeltaProgram,
    patch_tensors: Mapping[str, Tensor],
    checkpoint_tensors: Mapping[str, _StoredTensor],
    state_schema: ModelStateSchema | None,
) -> None:
    """Validate the complete program and state metadata before creating output."""

    targeted = set(program.targets)
    missing = targeted - set(checkpoint_tensors)
    if missing:
        raise ValueError(f"checkpoint lacks patch targets: {sorted(missing)}")
    alias_targets: set[str] = set()
    if state_schema is not None:
        schema_tensors = {
            specification.name: specification for specification in state_schema.tensors
        }
        for name in sorted(set(checkpoint_tensors).intersection(schema_tensors)):
            location = checkpoint_tensors[name]
            specification = schema_tensors[name]
            if location.shape != specification.shape or location.dtype != specification.dtype:
                raise ValueError(f"checkpoint tensor disagrees with state schema: {name}")
        for group in state_schema.aliases:
            selected = targeted.intersection(group.members)
            if selected and selected != set(group.members):
                missing_aliases = sorted(set(group.members) - selected)
                raise ValueError(f"tied parameter patch omits aliases: {missing_aliases}")
            alias_targets.update(selected)
    referenced: set[str] = set()
    alias_delta_hashes: dict[str, str] = {}
    with torch.no_grad():
        for target in sorted(program.targets):
            referenced.update(program.referenced_tensors(target))
            delta = program.materialize(target, patch_tensors)
            base = checkpoint_tensors[target]
            if delta.device.type != "cpu":
                raise ValueError(
                    f"checkpoint materialization requires a CPU delta tensor: {target}"
                )
            if state_schema is not None:
                specification = state_schema.tensor(target)
                if not specification.patchable:
                    raise ValueError(f"target is not patchable: {target}")
                if tuple(delta.shape) != specification.shape:
                    raise ValueError(f"target shape mismatch for {target}")
                if dtype_name(delta.dtype) != specification.dtype:
                    raise ValueError(f"target dtype mismatch for {target}")
            if tuple(delta.shape) != base.shape:
                raise ValueError(
                    f"base/delta shape mismatch for {target}: {base.shape} != {tuple(delta.shape)}"
                )
            if dtype_name(delta.dtype) != base.dtype:
                raise ValueError(
                    f"base/delta dtype mismatch for {target}: "
                    f"{base.dtype} != {dtype_name(delta.dtype)}"
                )
            if target in alias_targets:
                alias_delta_hashes[target] = tensor_content_hash(delta)
    unknown = referenced - set(patch_tensors)
    if unknown:
        raise ValueError(f"missing delta tensors: {sorted(unknown)}")
    unused = set(patch_tensors) - referenced
    if unused:
        raise ValueError(f"unreferenced delta tensors are not permitted: {sorted(unused)}")
    if state_schema is not None:
        for group in state_schema.aliases:
            if targeted.intersection(group.members):
                hashes = {alias_delta_hashes[member] for member in group.members}
                if len(hashes) != 1:
                    raise ValueError(f"inconsistent deltas for tied parameters: {group.members}")


def _load_output_shard(
    keys: tuple[str, ...],
    checkpoint_tensors: Mapping[str, _StoredTensor],
    measurements: _IOMeasurements,
) -> dict[str, Tensor]:
    """Load the physical source tensors needed for one planned output shard."""

    requested: dict[Path, dict[str, list[str]]] = {}
    for logical_key in keys:
        location = checkpoint_tensors[logical_key]
        requested.setdefault(location.path, {}).setdefault(location.storage_key, []).append(
            logical_key
        )
    result: dict[str, Tensor] = {}
    for path in sorted(requested):
        with safe_open(path, framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
            for storage_key in sorted(requested[path]):
                location = checkpoint_tensors[requested[path][storage_key][0]]
                started = time.perf_counter()
                physical = cast(Tensor, handle.get_tensor(storage_key)).clone()
                measurements.read_seconds += time.perf_counter() - started
                measurements.read_bytes += location.nbytes
                if (
                    tuple(physical.shape) != location.shape
                    or dtype_name(physical.dtype) != location.dtype
                    or physical.numel() * physical.element_size() != location.nbytes
                ):
                    raise RuntimeError(
                        f"checkpoint tensor metadata changed while reading: {storage_key}"
                    )
                logical_keys = sorted(requested[path][storage_key])
                result[logical_keys[0]] = physical
                for logical_key in logical_keys[1:]:
                    result[logical_key] = physical.clone()
    return dict(sorted(result.items()))


def _peak_rss() -> tuple[int | None, str]:
    """Return the process-lifetime RSS high-water mark when the platform exposes it."""

    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmHWM:"):
                    fields = line.split()
                    if len(fields) == 3 and fields[2] == "kB":
                        return int(fields[1]) * 1024, "linux_proc_status_vmhwm"
        except (OSError, UnicodeError, ValueError):
            pass
    try:
        import resource
    except ImportError:
        return None, "unavailable"
    getrusage = getattr(resource, "getrusage", None)
    rus_self = getattr(resource, "RUSAGE_SELF", None)
    if not callable(getrusage) or rus_self is None:
        return None, "unavailable"
    usage = getrusage(rus_self)
    maximum_rss = getattr(usage, "ru_maxrss", None)
    if not isinstance(maximum_rss, int | float):
        return None, "unavailable"
    scale = 1 if sys.platform == "darwin" else 1024
    return int(maximum_rss) * scale, "resource_getrusage_process_lifetime_peak"


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
    """Stream an additive program into a new checkpoint via an atomic directory rename."""

    operation_started = time.perf_counter()
    source = Path(source_checkpoint)
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    # Resolve parents without requiring the not-yet-created output path.
    source_resolved = source.resolve()
    target_resolved = target.parent.resolve() / target.name
    if source_resolved == target_resolved or source_resolved in target_resolved.parents:
        raise ValueError("output must not be the source checkpoint or nested within it")
    files = checkpoint_files(source)
    before_hashes = _source_file_hashes(source)
    measurements = _IOMeasurements()
    physical_tensors = _checkpoint_tensor_index(files)
    checkpoint_tensors = _expand_alias_index(physical_tensors, state_schema, measurements)
    _validate_preflight(program, patch_tensors, checkpoint_tensors, state_schema)
    tensor_sizes = {key: value.nbytes for key, value in checkpoint_tensors.items()}
    shards = plan_shards_by_size(tensor_sizes, max_shard_size=max_shard_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    output_files: list[str] = []
    output_tensor_hashes: dict[str, str] = {}
    weight_map: dict[str, str] = {}
    try:
        copy_started = time.perf_counter()
        auxiliary_files = _copy_auxiliary_files(source, temporary)
        copy_seconds = time.perf_counter() - copy_started
        auxiliary_bytes = sum((temporary / name).stat().st_size for name in auxiliary_files)
        measurements.read_bytes += auxiliary_bytes
        measurements.read_seconds += copy_seconds
        measurements.write_bytes += auxiliary_bytes
        measurements.write_seconds += copy_seconds
        shard_count = len(shards)
        for index, keys in enumerate(shards, start=1):
            filename = (
                "model.safetensors"
                if shard_count == 1
                else f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            )
            base_shard = _load_output_shard(keys, checkpoint_tensors, measurements)
            patched_shard: dict[str, Tensor] = {}
            with torch.no_grad():
                for key in keys:
                    base = base_shard.pop(key)
                    if key in program.targets:
                        delta = program.materialize(key, patch_tensors)
                        if tuple(delta.shape) != tuple(base.shape) or delta.dtype != base.dtype:
                            raise RuntimeError(f"preflight state changed while applying: {key}")
                        value = base + delta
                    else:
                        value = base
                    patched_shard[key] = value
                    output_tensor_hashes[key] = tensor_content_hash(value)
            write_started = time.perf_counter()
            save_safetensors_atomic(
                temporary / filename,
                patched_shard,
                metadata={
                    "format": "pt",
                    "modelpact_format": "modelpact-materialized-v1",
                },
                overwrite=False,
            )
            measurements.write_seconds += time.perf_counter() - write_started
            measurements.write_bytes += (temporary / filename).stat().st_size
            output_files.append(filename)
            weight_map.update(dict.fromkeys(keys, filename))
            del base_shard, patched_shard
        if shard_count > 1:
            index_value = {
                "metadata": {"total_size": sum(tensor_sizes.values())},
                "weight_map": dict(sorted(weight_map.items())),
            }
            write_started = time.perf_counter()
            atomic_write_text(
                temporary / "model.safetensors.index.json",
                canonical_dumps(index_value),
                overwrite=False,
            )
            measurements.write_seconds += time.perf_counter() - write_started
            measurements.write_bytes += (temporary / "model.safetensors.index.json").stat().st_size
            output_files.append("model.safetensors.index.json")
        after_hashes = _source_file_hashes(source)
        if after_hashes != before_hashes:
            raise RuntimeError("source checkpoint changed during materialization")
        peak_rss_bytes, peak_rss_method = _peak_rss()
        largest_tensor = max(tensor_sizes.values(), default=0)
        largest_shard = max((sum(tensor_sizes[key] for key in keys) for keys in shards), default=0)
        manifest: dict[str, object] = {
            "auxiliary_files": list(auxiliary_files),
            "output_files": sorted(output_files),
            "output_tensor_hashes": dict(sorted(output_tensor_hashes.items())),
            "performance": {
                "largest_output_shard_tensor_bytes": largest_shard,
                "largest_output_tensor_bytes": largest_tensor,
                "max_shard_size_bytes": max_shard_size,
                "measurement_scope": (
                    "tensor-payload reads (including alias validation), auxiliary copies, "
                    "and materialized checkpoint writes; source hashing and manifest writing "
                    "are excluded from read/write counters"
                ),
                "peak_rss_bytes": peak_rss_bytes,
                "peak_rss_method": peak_rss_method,
                "read_bytes": measurements.read_bytes,
                "read_seconds": measurements.read_seconds,
                "streaming_strategy": "planned-output-shard",
                "wall_seconds_before_manifest": time.perf_counter() - operation_started,
                "write_bytes": measurements.write_bytes,
                "write_seconds": measurements.write_seconds,
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
        os.replace(temporary, target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
