"""Atomic construction and hostile-input loading of patch directory bundles."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from torch import Tensor

from modelpact.models.schema import ModelStateSchema
from modelpact.patch.ast import DeltaProgram
from modelpact.patch.manifest import PatchManifest
from modelpact.patch.tensors import load_patch_tensors, save_patch_tensors
from modelpact.patch.validate import load_delta_program
from modelpact.util.atomic import atomic_write_bytes, atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import sha256_file

MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_SUPPLEMENTAL_ARTIFACT_BYTES = 512 * 1024**2
MANDATORY_BUNDLE_ARTIFACTS = frozenset(
    {
        "delta-program.json",
        "tensors.safetensors",
        "contracts/target.yaml",
        "contracts/preservation.yaml",
        "probes/manifest.json",
        "probes/hashes.json",
        "evidence/compile.json",
        "evidence/validation.json",
        "evidence/holdout.json",
        "evidence/minimization.json",
        "certificate.json",
        "report.md",
        "apply_patch.py",
        "verify_patch.py",
    }
)


def _safe_relative(path: str) -> str:
    normalized = path.replace("\\", "/")
    candidate = Path(normalized)
    if (
        not path
        or candidate.is_absolute()
        or bool(candidate.drive)
        or ".." in candidate.parts
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise ValueError(f"unsafe bundle path: {path}")
    return candidate.as_posix()


def _bundle_file(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    unresolved = root / Path(safe)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ValueError(f"bundle artifact must be a regular file: {safe}")
    resolved_root = root.resolve()
    resolved = unresolved.resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"bundle artifact escapes bundle: {safe}")
    return unresolved


def _supplemental_name(path: str) -> str:
    safe = _safe_relative(path)
    if safe.startswith(("probes/", "evidence/")) or safe in {
        "certificate.json",
        "report.md",
        "apply_patch.py",
        "verify_patch.py",
    }:
        return safe
    raise ValueError(f"unsupported supplemental patch artifact: {path}")


def missing_bundle_artifacts(manifest: PatchManifest) -> tuple[str, ...]:
    return tuple(sorted(MANDATORY_BUNDLE_ARTIFACTS - set(manifest.artifact_hashes)))


def require_complete_bundle(manifest: PatchManifest) -> None:
    missing = missing_bundle_artifacts(manifest)
    if missing:
        raise ValueError(f"patch bundle is incomplete; missing artifacts: {list(missing)}")


@dataclass(frozen=True, slots=True)
class PatchBundle:
    path: Path
    manifest: PatchManifest
    program: DeltaProgram
    tensors: Mapping[str, Tensor]


def create_patch_bundle(
    output: str | Path,
    *,
    name: str,
    base_signature: Mapping[str, object],
    state_schema: ModelStateSchema,
    program: DeltaProgram,
    tensors: Mapping[str, Tensor],
    tool_version: str,
    contracts: Mapping[str, bytes] | None = None,
    supplemental_artifacts: Mapping[str, bytes] | None = None,
    provides: tuple[str, ...] = (),
    preserves: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    verification_policy_hash: str | None = None,
    parent_patches: tuple[str, ...] = (),
    merged_from: tuple[str, ...] = (),
    rebased_from: str | None = None,
    source_diff_bundle: str | None = None,
    compiler_configuration: Mapping[str, object] | None = None,
    require_complete: bool = False,
) -> PatchBundle:
    """Create a content-addressed patch bundle without executing bundle content."""

    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    program.validate(tensors, state_schema)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        atomic_write_text(
            temporary / "delta-program.json", canonical_dumps(program.to_dict()), overwrite=False
        )
        save_patch_tensors(temporary / "tensors.safetensors", tensors)
        for relative, data in sorted((contracts or {}).items()):
            safe_name = _safe_relative(relative)
            if not safe_name.startswith("contracts/"):
                raise ValueError("contract artifacts must be below contracts/")
            atomic_write_bytes(temporary / safe_name, data, overwrite=False)
        supplemental_paths = []
        supplemental_size = 0
        for relative, data in sorted((supplemental_artifacts or {}).items()):
            safe_name = _supplemental_name(relative)
            supplemental_size += len(data)
            if supplemental_size > MAX_SUPPLEMENTAL_ARTIFACT_BYTES:
                raise ValueError("supplemental patch artifacts exceed the size limit")
            atomic_write_bytes(temporary / safe_name, data, overwrite=False)
            supplemental_paths.append(safe_name)
        artifact_paths = [
            "delta-program.json",
            "tensors.safetensors",
            *(contracts or {}).keys(),
            *supplemental_paths,
        ]
        artifact_hashes = {
            _safe_relative(relative): sha256_file(temporary / _safe_relative(relative))
            for relative in sorted(artifact_paths)
        }
        incomplete = PatchManifest(
            schema_version=1,
            tool_version=tool_version,
            patch_id="",
            name=name,
            base_signature=dict(base_signature),
            target_module_schema_hash=state_schema.schema_hash,
            delta_representation="additive_low_rank_sparse_v1",
            provides=tuple(sorted(set(provides))),
            preserves=tuple(sorted(set(preserves))),
            requires=tuple(sorted(set(requires))),
            artifact_hashes=artifact_hashes,
            verification_policy_hash=verification_policy_hash,
            parent_patches=tuple(sorted(set(parent_patches))),
            merged_from=tuple(sorted(set(merged_from))),
            rebased_from=rebased_from,
            source_diff_bundle=source_diff_bundle,
            compiler_configuration=dict(compiler_configuration or {}),
        )
        manifest = replace(incomplete, patch_id=incomplete.computed_patch_id())
        if require_complete:
            require_complete_bundle(manifest)
        atomic_write_text(
            temporary / "manifest.json", canonical_dumps(manifest.to_dict()), overwrite=False
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_patch_bundle(target, state_schema=state_schema)


def attach_bundle_artifacts(
    path: str | Path,
    artifacts: Mapping[str, bytes],
    *,
    state_schema: ModelStateSchema | None = None,
    require_complete: bool = False,
) -> PatchBundle:
    """Attach evidence/codegen after patch-ID creation and re-hash the manifest.

    Artifact files are written atomically and the manifest is updated last. An
    interrupted operation can leave an unreferenced file, but never a manifest
    that claims a partial or incorrectly hashed artifact.
    """

    bundle = load_patch_bundle(path, state_schema=state_schema)
    normalized: dict[str, bytes] = {}
    total_size = 0
    for relative, data in artifacts.items():
        safe = _supplemental_name(relative)
        if safe in normalized:
            raise ValueError(f"duplicate normalized supplemental artifact path: {safe}")
        if safe in bundle.manifest.artifact_hashes or (bundle.path / safe).exists():
            raise FileExistsError(bundle.path / safe)
        total_size += len(data)
        if total_size > MAX_SUPPLEMENTAL_ARTIFACT_BYTES:
            raise ValueError("supplemental patch artifacts exceed the size limit")
        normalized[safe] = data
    new_hashes = dict(bundle.manifest.artifact_hashes)
    for relative, data in sorted(normalized.items()):
        artifact_path = bundle.path / relative
        atomic_write_bytes(artifact_path, data, overwrite=False)
        new_hashes[relative] = sha256_file(artifact_path)
    manifest = replace(bundle.manifest, artifact_hashes=dict(sorted(new_hashes.items())))
    manifest.validate_identity()
    if require_complete:
        require_complete_bundle(manifest)
    atomic_write_text(bundle.path / "manifest.json", canonical_dumps(manifest.to_dict()))
    return load_patch_bundle(bundle.path, state_schema=state_schema)


def load_patch_bundle(
    path: str | Path, *, state_schema: ModelStateSchema | None = None
) -> PatchBundle:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"patch bundle must be a regular directory: {root}")
    manifest_path = _bundle_file(root, "manifest.json")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("patch manifest exceeds size limit")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("malformed patch manifest JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("patch manifest must be a JSON object")
    manifest = PatchManifest.from_dict(value)
    manifest.validate_identity()
    for relative, expected in manifest.artifact_hashes.items():
        actual = sha256_file(_bundle_file(root, relative))
        if actual != expected:
            raise ValueError(f"patch artifact hash mismatch: {relative}")
    program = load_delta_program(_bundle_file(root, "delta-program.json"))
    tensors = load_patch_tensors(_bundle_file(root, "tensors.safetensors"))
    program.validate(tensors, state_schema)
    if state_schema is not None and manifest.target_module_schema_hash != state_schema.schema_hash:
        raise ValueError("patch target module schema does not match loaded model")
    return PatchBundle(root, manifest, program, tensors)
