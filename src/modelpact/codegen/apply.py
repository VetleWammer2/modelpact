"""Emit the standalone Patch Bundle v1 application program."""

from __future__ import annotations

from pathlib import Path

from modelpact.codegen._emit import emit_script


def emit_apply_script(
    patch_bundle: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a package-independent, patch-ID-pinned application script."""

    return emit_script("apply", patch_bundle, output, overwrite=overwrite)


generate_apply_script = emit_apply_script

__all__ = ["emit_apply_script", "generate_apply_script"]
