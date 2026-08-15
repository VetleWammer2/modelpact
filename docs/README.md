# ModelPact documentation

The normative wire formats and claim rules live in `SPEC.md`; the research and
systems rationale lives in `TECHNICAL_NOTE.md`. The documents in this directory
explain operational boundaries without widening those specifications.

- [Concepts](concepts.md)
- [Adapters and trusted code](adapters.md)
- [Behavior contracts](contracts.md)
- [Patch format](patch-format.md)
- [Compiler](compiler.md)
- [Composition and semantic merge](composition.md)
- [Higher-order audit](audit.md)
- [Semantic rebase](rebase.md)
- [Verification certificates](certificates.md)
- [Trust model](trust-model.md)
- [Limitations](limitations.md)

## Stable executable commands

The committed CPU experiments use the module runner directly and work without
network access after installation:

```console
python -m modelpact.modelpactbench.runner closure_matrix --output artifacts/closure.json
python -m modelpact.modelpactbench.runner collusion --output artifacts/collusion.json
python -m modelpact.modelpactbench.runner merge --output artifacts/merge.json
python -m modelpact.modelpactbench.runner rebase --output artifacts/rebase.json
python -m modelpact.modelpactbench.runner rebase_cross_architecture --output artifacts/rebase-cross-architecture.json
python -m modelpact.modelpactbench.runner cegis --output artifacts/cegis.json
python -m modelpact.modelpactbench.runner forkbench --output artifacts/forkbench.json --artifacts artifacts/forkbench-run
```

The configuration-driven equivalent is:

```console
python benchmarks/run.py \
  --config benchmarks/closure_matrix/config.json \
  --output artifacts/closure-envelope.json
```

These commands generate new evidence. They do not replay committed terminal
text. CLI examples for model scanning, compilation, verification, composition,
audit, and rebase should be taken from `modelpact --help` for the installed
version; documentation does not guess flags that are not implemented.
