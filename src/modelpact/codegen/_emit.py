"""Shared implementation for emitting self-contained, data-only patch tools."""

from __future__ import annotations

import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Literal

from modelpact.codegen._template import STANDALONE_TEMPLATE
from modelpact.patch.bundle import load_patch_bundle

_TINY_ADAPTER_RESOURCE = "_tiny_adapter_runtime.py"
_TINY_ADAPTER_START = "# MODELPACT_BUILTIN_TINY_ADAPTER_BEGIN"
_TINY_ADAPTER_END = "# MODELPACT_BUILTIN_TINY_ADAPTER_END"
_HUGGINGFACE_ADAPTER_RESOURCE = "_huggingface_adapter_runtime.py"
_HUGGINGFACE_ADAPTER_START = "# MODELPACT_BUILTIN_HUGGINGFACE_ADAPTER_BEGIN"
_HUGGINGFACE_ADAPTER_END = "# MODELPACT_BUILTIN_HUGGINGFACE_ADAPTER_END"


def _builtin_tiny_adapter_source() -> str:
    source = files("modelpact.codegen").joinpath(_TINY_ADAPTER_RESOURCE).read_text("utf-8")
    try:
        fragment = source.split(_TINY_ADAPTER_START, 1)[1].split(_TINY_ADAPTER_END, 1)[0]
    except IndexError as error:
        raise RuntimeError("standalone tiny adapter source markers are missing") from error
    if len(fragment.encode("utf-8")) > 1024 * 1024:
        raise RuntimeError("standalone tiny adapter source exceeds its size limit")
    return fragment.strip() + "\n"


def _builtin_huggingface_adapter_source() -> str:
    source = files("modelpact.codegen").joinpath(_HUGGINGFACE_ADAPTER_RESOURCE).read_text("utf-8")
    try:
        fragment = source.split(_HUGGINGFACE_ADAPTER_START, 1)[1].split(
            _HUGGINGFACE_ADAPTER_END, 1
        )[0]
    except IndexError as error:
        raise RuntimeError("standalone Hugging Face adapter source markers are missing") from error
    if len(fragment.encode("utf-8")) > 1024 * 1024:
        raise RuntimeError("standalone Hugging Face adapter source exceeds its size limit")
    return fragment.strip() + "\n"


def emit_script(
    mode: Literal["apply", "verify"],
    patch_bundle: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
    will_live_in_bundle: bool = False,
) -> Path:
    """Emit a script pinned to a validated Patch Bundle v1 identity.

    The generated source contains no import of :mod:`modelpact`.  The bundle is
    still supplied as data at execution time (and can be relocated with
    ``--patch``); the embedded patch ID prevents silently substituting another
    bundle.
    """

    bundle = load_patch_bundle(patch_bundle)
    target = Path(output)
    if target.is_symlink():
        raise ValueError("refusing to replace a symlink with a generated script")
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if will_live_in_bundle:
        default_patch_relative: str | None = "."
    else:
        try:
            # ``resolve()`` can return an 8.3 path for one side and a long path for
            # the other on Windows.  Detect the common in-bundle emission case by
            # file identity before computing a lexical relative path.
            relative_bundle = (
                Path(".")
                if os.path.samefile(bundle.path, target.parent)
                else Path(os.path.relpath(bundle.path.resolve(), start=target.parent.resolve()))
            )
            default_patch_relative = PurePosixPath(*relative_bundle.parts).as_posix()
        except ValueError:
            # Paths on different Windows drives have no portable relative form. The
            # tool remains usable with ``--patch`` without leaking a build path.
            default_patch_relative = None
    source = (
        STANDALONE_TEMPLATE.replace("@@MODE@@", json.dumps(mode))
        .replace("@@EXPECTED_PATCH_ID@@", json.dumps(bundle.manifest.patch_id))
        .replace("@@EXPECTED_EVIDENCE_ID@@", json.dumps(bundle.evidence_id))
        .replace("@@DEFAULT_PATCH_RELATIVE@@", json.dumps(default_patch_relative))
        .replace(
            "@@BUILTIN_TINY_ADAPTER@@",
            _builtin_tiny_adapter_source() if mode == "verify" else "",
        )
        .replace(
            "@@BUILTIN_HUGGINGFACE_ADAPTER@@",
            _builtin_huggingface_adapter_source() if mode == "verify" else "",
        )
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".py", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(source, encoding="utf-8", newline="\n")
        if target.is_symlink():
            raise ValueError("refusing to replace a symlink with a generated script")
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
