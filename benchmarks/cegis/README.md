# Locality and CEGIS

The CPU organism fits a differentiable polynomial behavior patch, searches a
finite neighborhood for failures, adds executed counterexamples, and recompiles.
It deliberately retains its negative case when the bounded loop does not remove
every search failure.

```console
python benchmarks/run.py \
  --config benchmarks/cegis/config.json \
  --output artifacts/pactbench/cegis.json
```

Report both the initial and post-CEGIS failure counts. A bounded search with no
new failure would still not be a universal guarantee.
