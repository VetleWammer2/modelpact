# Semantic merge example

This example uses the same executable configuration schema as ModelPactBench. It
demonstrates the distinction between additive composition and a semantic merge:
the former only sums and verifies, while the latter runs a new optimization
against the parent contracts.

```console
python benchmarks/run.py \
  --config examples/semantic_merge/config.json \
  --output artifacts/examples/semantic-merge.json
```

The generated JSON records the naive closure result, whether the compiler was
invoked, optimization steps, and merged verification. It is an analytic CPU
example and makes no language-model baseline claim.
