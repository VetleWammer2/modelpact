# Contributing to ModelPact

ModelPact is a research system whose claims must remain narrower than its
evidence. Contributions are welcome when they preserve that standard and the
data/code trust boundary.

## Development setup

Use Python 3.11 or newer on Linux for release validation. CPU development is
supported; an NVIDIA GPU is optional for larger compilation experiments.

```console
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install ".[dev,research]"
```

Normal tests must not download models or contact remote inference APIs. Hugging
Face integration tests use a checkpoint explicitly supplied by the operator and
must keep `local_files_only=True`, `trust_remote_code=False`, and SafeTensors
loading enabled.

## Required checks

Run the same gates as CPU CI:

```console
python -m ruff format --check .
python -m ruff check .
python -m mypy src/modelpact
python -m pytest
python -m build
```

Run the fast real PactBench experiments when changing compilation, composition,
audit, merge, or rebase logic:

```console
python benchmarks/run.py --config benchmarks/closure_matrix/config.json --output artifacts/closure.json
python benchmarks/run.py --config benchmarks/collusion/config.json --output artifacts/collusion.json
python benchmarks/run.py --config benchmarks/merge/config.json --output artifacts/merge.json
python benchmarks/run.py --config benchmarks/rebase/config.json --output artifacts/rebase.json
python benchmarks/run.py --config benchmarks/cegis/config.json --output artifacts/cegis.json
```

Generated outputs belong under ignored `artifacts/` unless a research review
deliberately curates the raw result, configuration, and environment manifest.

## Trust and security

Model adapters are explicitly trusted local Python. Contracts, patch bundles,
delta programs, certificates, checkpoints, probe manifests, and lockfiles are
untrusted data. Changes handling those artifacts must retain strict schemas,
resource limits, path containment, symlink rejection, atomic writes, and the
prohibition on pickle, `eval`, serialized callables, and shell-string execution.

Security tests should mutate hashes, tensor bytes, contracts, tokenizers, base
checkpoints, and paths. A failed or unsupported check must never become a pass by
default.

## Research changes

- Separate compile objectives from acceptance assertions.
- Keep compile/search/validation/guard/sealed-holdout roles distinct.
- Open a sealed holdout only under its declared final-candidate or independent
  verification policy. A failed holdout ends that contract version.
- Retain unsuccessful runs, counterexamples, unresolved compositions, rebase
  failures, and baselines that win.
- Report raw prompt-level distributions and deterministic seeds. One seed is not
  evidence of superiority or statistical significance.
- Call an audit exhaustive only when every relevant subset was executed.
- Use only local or budgeted minimality wording unless the complete search space
  was evaluated.

New benchmark behavior names belong in benchmark data and reports, not in
production extraction or compiler logic.

## Pull requests

Keep commits logically scoped. A pull request should state the exact commands
run, CPU/GPU environment, generated artifact locations, negative results, and
unsupported cases. Never edit a transcript to look better or infer a successful
result from a mock. Documentation claims must point to committed machine-readable
evidence or be clearly marked as design, limitation, or unexecuted protocol.

By contributing, you agree that your contribution is provided under the
repository's Apache-2.0 license.
