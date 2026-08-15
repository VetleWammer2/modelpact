# Benign Collusion

This experiment executes individually passing scalar patches whose relevant
pairs pass while a harmless three-way combination violates a required-output
margin. The active auditor proposes and executes subsets; when budget remains,
ddmin executes additional candidates to attempt reduction. The surrogate never
supplies an outcome itself.

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
