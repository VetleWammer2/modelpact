from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def test_retained_summary_matches_readme_and_fails_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "benchmarks" / "summarize.py")]
    completed = subprocess.run(  # noqa: S603 - exact interpreter and repository script
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    readme = (root / "README.md").read_text(encoding="utf-8")
    transcript = re.search(
        r"\$ python benchmarks/summarize\.py\n(.*?)\n```",
        readme,
        flags=re.DOTALL,
    )
    assert transcript is not None
    assert transcript.group(1).strip() == completed.stdout.strip()

    mutated = tmp_path / "artifacts"
    shutil.copytree(root / "research" / "artifacts", mutated)
    fork_path = mutated / "forkbench.json"
    fork = json.loads(fork_path.read_text(encoding="utf-8"))
    fork["success"] = False
    fork["status"] = "FAIL"
    fork_path.write_text(json.dumps(fork), encoding="utf-8")
    failed = subprocess.run(  # noqa: S603 - exact interpreter and repository script
        [*command, str(mutated)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert failed.returncode != 0
    assert "not a successful terminal result" in failed.stderr
