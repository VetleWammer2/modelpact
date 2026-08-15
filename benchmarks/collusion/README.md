# Benign Collusion

This experiment executes six independently compiled low-rank Behavior Patch
bundles on TinyCausalLM. Every singleton and pair retains a required synthetic
output token, while one harmless three-patch combination flips it. The active
auditor proposes and executes real model stacks; ddmin executes additional
stacks to reduce a witnessed failure. The surrogate never supplies an outcome
itself.

```console
python benchmarks/run.py \
  --config benchmarks/collusion/config.json \
  --output artifacts/pactbench/collusion.json
```

The committed run is an `ACTIVE_BUDGETED` audit and must not be described as
exhaustive. Its executed-subset count, failures, reduction result, and
`budget_exhausted` flag are authoritative for each generated artifact. A found
failure does not imply that every other subset was evaluated, and an unreduced
failure is not a minimal-failing-subset claim.

The result also reports executed singleton-only, pairwise-only, deterministic
random, and parameter-overlap baselines. Parameter overlap is only a selection
heuristic; all six patches touch the same output matrix, so it is not treated as
a compatibility verdict.
