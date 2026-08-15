# Local Hugging Face preflight

This runner performs real local checkpoint loading, complete ModelPact model
fingerprinting, and autoregressive generation. It never downloads a model,
enables remote model code, or accepts a repository identifier in place of a
local directory. Third-party weights are not committed.

Install the optional dependency and point the runner at a decoder-only causal
LM stored locally in SafeTensors format:

```console
python -m pip install ".[huggingface]"
python benchmarks/huggingface_local/run.py \
  --checkpoint /absolute/path/to/local-checkpoint \
  --output artifacts/pactbench/huggingface-local.json
```

The output contains tensor/tokenizer identity and hashes of generated samples.
It intentionally emits no patch-compilation, composition, extraction, rebase,
or compatibility claim. A full PactBench G result additionally requires two
verified behavior patches, composition, rebase or extraction, and their evidence
bundles; this preflight does not substitute for that experiment.
