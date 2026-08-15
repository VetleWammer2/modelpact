# Semantic Merge

The analytic merge case executes two parent deltas independently, executes their
naive sum, then invokes a new constrained PyTorch optimization when the sum is
not contract-closed. The result records whether the compiler ran and whether
the newly optimized delta passed the union margins.

```console
python benchmarks/run.py \
  --config benchmarks/merge/config.json \
  --output artifacts/pactbench/semantic-merge.json
```

This is an exact systems test for the merge control flow, not evidence that the
method outperforms model-merging baselines on language models.
