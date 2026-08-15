# RebaseBench analytic cases

This configuration runs both the same-family and controlled cross-architecture
rebase organisms. The same-family path executes and rejects an invalid direct
transfer before recompiling. The cross-architecture path skips tensor
transplantation and compiles behavior against compatible input/output semantics.

```console
python benchmarks/run.py \
  --config benchmarks/rebase/config.json \
  --output artifacts/pactbench/rebase.json
```

The output reports direct-transfer and recompilation evidence separately. It
does not establish compatibility for architectures beyond the executed cases.
