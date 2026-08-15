from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import modelpact.cli as cli_module
from modelpact.cli import _read_lock, _verify_locked_patch_manifests, app
from modelpact.compose.stack import MAX_STACK_LOCK_PATCHES, StackLock
from modelpact.patch.bundle import MAX_MANIFEST_BYTES
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import sha256_file

RUNNER = CliRunner()


def _digest(index: int) -> str:
    return f"sha256:{index:064x}"


def _lock_fixture(tmp_path: Path) -> tuple[Path, Path, str, dict[str, object]]:
    patch_id = _digest(1)
    patch = tmp_path / "patch"
    patch.mkdir()
    manifest = patch / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    resolved = tmp_path / "resolved"
    value: dict[str, object] = {
        "audit_hash": None,
        "base_hash": _digest(2),
        "certificate_hash": None,
        "contract_hashes": [_digest(3)],
        "extensions": {
            "modelpact_cli": {
                "base_manifest_hash": _digest(4),
                "base_path": base.resolve().as_posix(),
                "dependency_order": [patch_id],
                "patch_paths": {patch_id: patch.resolve().as_posix()},
                "resolved_patch_path": resolved.resolve().as_posix(),
            }
        },
        "patch_hashes": {patch_id: sha256_file(manifest)},
        "resolution": "NAIVE_ADDITIVE_STACK",
        "resolved_artifact_hash": _digest(5),
        "schema_version": 1,
        "verification_policy_hash": _digest(6),
    }
    lock = tmp_path / "stack.lock.json"
    lock.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
    return lock, manifest, patch_id, value


def _write_variant(tmp_path: Path, index: int, value: dict[str, object]) -> Path:
    path = tmp_path / f"invalid-{index}.lock.json"
    path.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
    return path


def test_stack_lock_parser_rejects_unknown_malformed_and_inconsistent_data(
    tmp_path: Path,
) -> None:
    _lock, _manifest, patch_id, original = _lock_fixture(tmp_path)
    variants: list[dict[str, object]] = []

    unknown = copy.deepcopy(original)
    unknown["unexpected"] = True
    variants.append(unknown)

    bad_hash = copy.deepcopy(original)
    bad_hash["base_hash"] = "not-a-digest"
    variants.append(bad_hash)

    bad_resolution = copy.deepcopy(original)
    bad_resolution["resolution"] = "CLOSED"
    variants.append(bad_resolution)

    missing_success_artifact = copy.deepcopy(original)
    missing_success_artifact["resolved_artifact_hash"] = None
    variants.append(missing_success_artifact)

    duplicate_contract = copy.deepcopy(original)
    duplicate_contract["contract_hashes"] = [_digest(3), _digest(3)]
    variants.append(duplicate_contract)

    unknown_extension = copy.deepcopy(original)
    extensions = unknown_extension["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["unexpected"] = True
    variants.append(unknown_extension)

    relative_path = copy.deepcopy(original)
    extensions = relative_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["base_path"] = "relative/base"
    variants.append(relative_path)

    network_path = copy.deepcopy(original)
    extensions = network_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["base_path"] = "//server/share/model"
    variants.append(network_path)

    null_extension = copy.deepcopy(original)
    null_extension["extensions"] = {"modelpact_cli": None}
    variants.append(null_extension)

    missing_patch_path = copy.deepcopy(original)
    extensions = missing_patch_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["patch_paths"] = {}
    variants.append(missing_patch_path)

    missing_dependency = copy.deepcopy(original)
    extensions = missing_dependency["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["dependency_order"] = []
    variants.append(missing_dependency)

    too_many_patches = copy.deepcopy(original)
    too_many_patches["patch_hashes"] = {
        _digest(index + 100): _digest(10) for index in range(MAX_STACK_LOCK_PATCHES + 1)
    }
    variants.append(too_many_patches)

    for index, value in enumerate(variants):
        with pytest.raises(ValueError):
            _read_lock(_write_variant(tmp_path, index, value))

    parsed = _read_lock(_lock)
    assert isinstance(parsed.lock, StackLock)
    assert set(parsed.lock.patch_hashes) == {patch_id}


def test_locked_manifest_size_is_rejected_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest, _patch_id, value = _lock_fixture(tmp_path)
    with manifest.open("wb") as stream:
        stream.truncate(MAX_MANIFEST_BYTES + 1)
    lock.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
    parsed = _read_lock(lock)

    def forbidden_hash(_path: str | Path, *, max_bytes: int | None = None) -> str:
        del max_bytes
        raise AssertionError("oversized manifest must be rejected before hashing")

    monkeypatch.setattr(cli_module, "sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="exceeds the size limit"):
        _verify_locked_patch_manifests(parsed)


def test_locked_manifest_hash_is_explicitly_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, _manifest, _patch_id, _value = _lock_fixture(tmp_path)
    parsed = _read_lock(lock)
    calls: list[int | None] = []

    def bounded_hash(path: str | Path, *, max_bytes: int | None = None) -> str:
        calls.append(max_bytes)
        return sha256_file(path, max_bytes=max_bytes)

    monkeypatch.setattr(cli_module, "sha256_file", bounded_hash)
    _verify_locked_patch_manifests(parsed)
    assert calls == [MAX_MANIFEST_BYTES]


def test_revert_rejects_nonregular_manifest_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, manifest, patch_id, _value = _lock_fixture(tmp_path)
    manifest.unlink()
    manifest.mkdir()
    model_loaded = False

    def forbidden_model_load(*_args: object, **_kwargs: object) -> object:
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("the base model must not load before manifest preflight")

    monkeypatch.setattr(cli_module, "_load_model", forbidden_model_load)
    result = RUNNER.invoke(
        app,
        [
            "revert",
            str(lock),
            "--remove",
            patch_id,
            "--output",
            str(tmp_path / "output"),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert model_loaded is False
    payload = json.loads(result.stdout)
    assert payload["status"] == "ERROR"
    assert "regular file" in payload["error"]


def test_locked_symlink_manifest_is_rejected(tmp_path: Path) -> None:
    lock, manifest, _patch_id, value = _lock_fixture(tmp_path)
    target = tmp_path / "outside-manifest.json"
    target.write_text("{}\n", encoding="utf-8")
    manifest.unlink()
    try:
        manifest.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    patch_hashes = value["patch_hashes"]
    assert isinstance(patch_hashes, dict)
    patch_hashes[next(iter(patch_hashes))] = sha256_file(target)
    lock.write_text(canonical_dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="regular file"):
        _verify_locked_patch_manifests(_read_lock(lock))
