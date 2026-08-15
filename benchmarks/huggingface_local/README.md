# Local Hugging Face integration

Benchmark G has two offline modes. The generated-fixture mode executes the full
research path against a real `GPTNeoXForCausalLM`; the operator-supplied mode is
a lower-cost manifest and generation preflight for a checkpoint already on
disk. Neither mode downloads weights or enables remote model code.

## Full generated workflow

Install the optional dependencies and run:

```console
python -m pip install ".[huggingface]"
python benchmarks/huggingface_local/run.py \
  --generate-fixture \
  --output artifacts/pactbench/huggingface-local
```

This deterministically constructs a WordLevel tokenizer and a one-layer
GPT-NeoX model, trains a base, two single-change teachers, and a changed base-v2,
then reloads each checkpoint through ModelPact's local-only adapter. It compiles
two low-rank behavior patches, verifies target and preservation contracts under
free generation and a sealed holdout, executes additive contract closure, and
performs a direct-first semantic rebase. The output directory contains:

```text
huggingface-local/
├── generated-checkpoints/   # ephemeral, locally trained SafeTensors only
├── patch-fact-a/            # complete content-addressed patch bundle
├── patch-fact-b/            # complete content-addressed patch bundle
├── composition.json
├── rebase.json
└── result.json
```

Do not commit the generated checkpoints. The default `.gitignore` excludes the
recommended `artifacts/` output root. Results cover only the finite generated
fixture and one deterministic seed; they are not compatibility claims for
arbitrary Hugging Face model families.

## Operator-supplied checkpoint preflight

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

The preflight output contains tensor/tokenizer identity and hashes of generated
samples. It intentionally emits no patch-compilation, composition, extraction,
rebase, or compatibility claim; use `--generate-fixture` for the complete
Benchmark G experiment. Third-party model weights are never committed.
