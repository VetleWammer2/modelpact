"""Path validation for untrusted bundle-relative paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafePathError(ValueError):
    pass


def safe_relative_path(value: str, *, max_length: int = 4096) -> PurePosixPath:
    if not value or len(value) > max_length or "\x00" in value:
        raise UnsafePathError("path is empty, too long, or contains NUL")
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    windows_candidate = PureWindowsPath(value)
    if (
        candidate.is_absolute()
        or candidate.drive
        or windows_candidate.is_absolute()
        or bool(windows_candidate.drive)
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise UnsafePathError(f"unsafe relative path: {value!r}")
    return candidate


def resolve_inside(root: str | Path, relative: str) -> Path:
    base = Path(root).resolve()
    candidate = (base / Path(*safe_relative_path(relative).parts)).resolve()
    if candidate != base and base not in candidate.parents:
        raise UnsafePathError(f"path escapes bundle root: {relative!r}")
    return candidate
