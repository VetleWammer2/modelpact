"""Shared implementation for emitting self-contained, data-only patch tools."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from modelpact.codegen._template import STANDALONE_TEMPLATE
from modelpact.patch.bundle import load_patch_bundle


def emit_script(
    mode: Literal["apply", "verify"],
    patch_bundle: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
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
    source = (
        STANDALONE_TEMPLATE.replace("@@MODE@@", json.dumps(mode))
        .replace("@@EXPECTED_PATCH_ID@@", json.dumps(bundle.manifest.patch_id))
        .replace("@@DEFAULT_PATCH@@", json.dumps(str(bundle.path.resolve())))
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
