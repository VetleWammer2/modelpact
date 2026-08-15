# Model adapters

Adapters are the narrow trusted-code boundary between ModelPact and a local
model. They load a checkpoint, expose tokenizer semantics, deterministic logits
and generation, patchable modules, activation points, and a complete state
schema. Built-ins cover the internal tiny decoder and safe local Hugging Face
causal LMs. A custom adapter is referenced as `module:attribute`; importing it
executes trusted local Python.

The built-in Hugging Face adapter uses local-only loading and disables remote
code. A custom adapter can perform arbitrary actions, including network access;
the trusted adapter author is responsible for disclosing that behavior and for
exposing persistent state correctly. Checkpoint tensors, the state/module schema,
aliases, an allowlisted set of tokenizer files, chat template, generation
defaults, architecture configuration, and adapter identity participate in the
emitted model signature.
