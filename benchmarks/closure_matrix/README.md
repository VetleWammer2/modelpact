# Contract Closure Matrix

This configuration executes the six-patch analytic closure organism and all 63
nonempty subsets. Every margin is produced by a PyTorch forward pass. The run is
small enough for normal CPU CI and needs no network access.

From an installed checkout:

```console
python benchmarks/run.py \
  --config benchmarks/closure_matrix/config.json \
  --output artifacts/pactbench/closure-matrix.json
```

The output is generated evidence, not a committed release score. Inspect
`results.closure_matrix.claims`, `failing_subsets`, and
`search_space_exhausted`; only the executed 63-subset run supports exhaustive
wording for this particular analytic pool.
