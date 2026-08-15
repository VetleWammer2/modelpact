# ForkBench

ForkBench trains a deterministic tiny causal LM and a target checkpoint with
five learned changes, discovers scoped difference witnesses, selects one cluster,
compiles a low-rank patch, runs bounded CEGIS and minimization, and opens sealed
target and guard holdouts only for the selected final patch identity.

Run the configuration-only result path with:

```console
python benchmarks/run.py \
  --config benchmarks/forkbench/config.json \
  --output artifacts/forkbench.json
```

Retain the complete checkpoints, diff bundle, patch bundle, certificate, probe
manifests, standalone scripts, and raw result with:

```console
python -m modelpact.modelpactbench.runner forkbench \
  --output artifacts/forkbench/result.json \
  --artifacts artifacts/forkbench/run
```

The runner refuses to overwrite an existing retained artifact tree. Reported
claims apply only to the committed finite probes, deterministic mutation search,
generation policy, environment, and declared budgets.
