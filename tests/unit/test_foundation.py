from __future__ import annotations

import math

import pytest

from modelpact.status import VerificationOutcome
from modelpact.util.canonical_json import CanonicalJSONError, canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_bytes
from modelpact.util.paths import UnsafePathError, safe_relative_path


def test_canonical_json_is_order_independent() -> None:
    left = {"b": [2, 1], "a": "å"}
    right = {"a": "å", "b": [2, 1]}
    assert canonical_dumps(left) == canonical_dumps(right)
    assert hash_canonical(left) == hash_canonical(right)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_canonical_json_rejects_non_finite(value: float) -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_dumps({"value": value})


def test_hashes_are_algorithm_tagged() -> None:
    assert sha256_bytes(b"modelpact").startswith("sha256:")
    assert len(sha256_bytes(b"modelpact")) == 71


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:\\escape", "a/../../b", "a\\..\\b"])
def test_untrusted_paths_cannot_escape(path: str) -> None:
    with pytest.raises(UnsafePathError):
        safe_relative_path(path)


def test_inconclusive_is_not_success() -> None:
    assert VerificationOutcome.INCONCLUSIVE is not VerificationOutcome.PASS

