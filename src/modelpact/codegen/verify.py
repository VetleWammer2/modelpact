"""Emit the standalone Patch Bundle v1 independent verifier."""

from __future__ import annotations

from pathlib import Path

from modelpact.codegen._emit import emit_script


def emit_verify_script(
    patch_bundle: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
    will_live_in_bundle: bool = False,
) -> Path:
    """Write a package-independent verifier pinned to the selected patch ID."""

    return emit_script(
        "verify",
        patch_bundle,
        output,
        overwrite=overwrite,
        will_live_in_bundle=will_live_in_bundle,
    )


generate_verify_script = emit_verify_script

__all__ = ["emit_verify_script", "generate_verify_script"]
