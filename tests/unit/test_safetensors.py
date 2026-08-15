from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import torch
from safetensors import safe_open

from modelpact.checkpoints.safetensors import load_safetensors, save_safetensors_atomic
from modelpact.util.hashing import sha256_file


def test_atomic_safetensors_writes_are_byte_deterministic(tmp_path: Path) -> None:
    tensors = {
        "zeta": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "alpha": torch.tensor([7, 8, 9], dtype=torch.int64),
        "middle": torch.arange(5, dtype=torch.int16),
    }
    metadata = {
        "zeta": "last",
        "alpha": "first",
        "unicode": "på",
        "modelpact_format": "determinism-regression-v1",
    }

    paths = [tmp_path / f"repeat-{index}.safetensors" for index in range(16)]
    for path in paths:
        save_safetensors_atomic(path, tensors, metadata=metadata, overwrite=False)

    expected_bytes = paths[0].read_bytes()
    expected_hash = sha256_file(paths[0])
    assert all(path.read_bytes() == expected_bytes for path in paths[1:])
    assert {sha256_file(path) for path in paths} == {expected_hash}

    header_size = int.from_bytes(expected_bytes[:8], byteorder="little", signed=False)
    header = json.loads(expected_bytes[8 : 8 + header_size].decode("utf-8"))
    assert header["__metadata__"] == metadata
    assert expected_bytes[8 : 8 + header_size].rstrip(b" ") == json.dumps(
        header,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    offsets = sorted(tuple(header[key]["data_offsets"]) for key in tensors)
    assert offsets[0][0] == 0
    assert all(left[1] == right[0] for left, right in pairwise(offsets))
    assert offsets[-1][1] == len(expected_bytes) - 8 - header_size

    loaded = load_safetensors(paths[-1])
    assert loaded.keys() == tensors.keys()
    assert all(torch.equal(loaded[key], value) for key, value in tensors.items())
    with safe_open(paths[-1], framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
        assert handle.metadata() == metadata
