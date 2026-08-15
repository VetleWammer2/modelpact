# Contract Closure Matrix

This configuration deterministically builds one TinyCausalLM base and six
content-addressed low-rank Behavior Patch bundles, verifies every singleton,
and executes all 63 nonempty stacks. Every outcome comes from a runtime-mounted
delta, TinyCausalLM forward logits, and one-token autoregressive generation.
The run is small enough for normal CPU CI and needs no network access.

From an installed checkout:

```console
python benchmarks/run.py \
  --config benchmarks/closure_matrix/config.json \
  --output artifacts/pactbench/closure-matrix.json
```

The output is generated evidence, not a committed release score. Inspect
`results.closure_matrix.claims`, `failing_subsets`, and
`search_space_exhausted`; only the executed 63-subset run supports exhaustive
wording for this particular finite TinyCausalLM pool. `subset_ground_truth`
retains every signed contract margin, generated token identity, and execution
hash.
