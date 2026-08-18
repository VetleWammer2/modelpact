"""Path validation for untrusted bundle-relative paths."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafePathError(ValueError):
    pass


_WINDOWS_RESERVED_NAMES = frozenset(
    {"aux", "con", "nul", "prn"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"|?*')


def safe_relative_path(value: str, *, max_length: int = 4096) -> PurePosixPath:
    if not value or len(value) > max_length or "\x00" in value:
        raise UnsafePathError("path is empty, too long, or contains NUL")
    if "\\" in value:
        raise UnsafePathError("path must use canonical POSIX separators")
    candidate = PurePosixPath(value)
    windows_candidate = PureWindowsPath(value)
    unsafe_component = any(
        part.endswith((".", " "))
        or part.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        or any(character in _WINDOWS_FORBIDDEN_CHARACTERS for character in part)
        or any(ord(character) < 32 for character in part)
        for part in candidate.parts
    )
    if (
        candidate.is_absolute()
        or candidate.drive
        or windows_candidate.is_absolute()
        or bool(windows_candidate.drive)
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != value
        or unsafe_component
    ):
        raise UnsafePathError(f"unsafe relative path: {value!r}")
    return candidate


def validate_relative_paths(
    values: Iterable[str],
    *,
    reserved_paths: Iterable[str] = (),
) -> None:
    """Require unique portable spellings for an untrusted relative-path set."""

    reserved = {item.casefold(): item for item in reserved_paths}
    seen: dict[str, str] = {}
    for value in values:
        safe_relative_path(value)
        folded = value.casefold()
        prior = seen.get(folded)
        if prior is not None and prior != value:
            raise UnsafePathError(f"paths collide on a case-insensitive filesystem: {prior!r}")
        expected = reserved.get(folded)
        if expected is not None and value != expected:
            raise UnsafePathError(f"reserved path must use canonical spelling: {expected!r}")
        seen[folded] = value


def resolve_inside(root: str | Path, relative: str) -> Path:
    base = Path(root).resolve()
    candidate = (base / Path(*safe_relative_path(relative).parts)).resolve()
    if candidate != base and base not in candidate.parents:
        raise UnsafePathError(f"path escapes bundle root: {relative!r}")
    return candidate
