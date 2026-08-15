# Semantic rebase example

This example executes same-family and cross-architecture analytic rebase paths.
The direct same-family transplant is tested rather than assumed; a failed check
triggers behavioral recompilation. The cross-architecture case never copies
physical tensors between architectures.

```console
python benchmarks/run.py \
  --config examples/semantic_rebase/config.json \
  --output artifacts/examples/semantic-rebase.json
```

The resulting claims are scoped to the configured teachers, target/guard
margins, and optimization budget.
