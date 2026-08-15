# Model adapters

Adapters are the narrow trusted-code boundary between ModelPact and a local
model. They load a checkpoint, expose tokenizer semantics, deterministic logits
and generation, patchable modules, activation points, and a complete state
schema. Built-ins cover the internal tiny decoder and safe local Hugging Face
causal LMs. A custom adapter is referenced as `module:attribute`; importing it
executes trusted local Python.

Adapters must not hide remote downloads, enable remote code, or omit persistent
state. Tokenizer, chat-template, generation defaults, aliases, and checkpoint
tensors all participate in model identity.

