"""Record the local benchmark execution environment as canonical JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": Path(sys.executable).name,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        },
        "packages": {
            name: version
            for name in (
                "modelpact",
                "numpy",
                "pyarrow",
                "PyYAML",
                "safetensors",
                "scikit-learn",
                "scipy",
                "tokenizers",
                "transformers",
                "typer",
            )
            if (version := _version(name)) is not None
        },
        "gpu_execution_performed": False,
    }
    atomic_write_text(arguments.output, canonical_dumps(payload) + "\n")


if __name__ == "__main__":
    main()
