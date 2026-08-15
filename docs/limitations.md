# Limitations

- Contract verification covers declared finite, sampled, and searched spaces;
  passing is not universal safety or complete behavior preservation.
- Search can miss brittle prompts and higher-order interactions. Budgeted audit
  is never exhaustive.
- Optimization can be infeasible, unstable, or too expensive; a compact patch
  may not exist for a desired behavior.
- Greedy module/rank pruning supports only local or budget minimality claims.
- Numerical reproduction can vary across PyTorch/CUDA/hardware environments.
- R1 supports local decoder-only causal LMs and trusted PyTorch adapters. It does
  not support closed APIs, multimodal/diffusion, RLHF, quantized training, MoE,
  arbitrary tensor parallelism, remote registries, signing, or formal open-ended
  verification.
- Same-family rebase still requires executed evidence. Cross-architecture rebase
  is behavioral recompilation and requires compatible tokenizer/output semantics.
- Automatic compilation currently emits low-rank deltas only for linear weights.
  Sparse/vector IR operations are mount/materialization capabilities, not
  automatically synthesized residuals.
- CEGIS exposes a real callback-driven loop, but `compile` cannot invent a
  search space: positive CLI CEGIS rounds require an explicit search backend and
  otherwise return `UNSUPPORTED`.
- The CLI provides joint merge and semantic rebase compilation for the built-in
  tiny adapter. Custom and Hugging Face repair/recompile paths require an
  explicit trusted compiler integration; tiny rebase currently accepts one
  deduplicated guarded contract. A repaired `merge` output is an executed
  composition artifact, not a complete Behavior Patch Bundle v1.
- Greedy minimization tests individual module removals and rank truncations; it
  does not test module groups, reoptimize candidates, or prune sparse residuals.
- Checkpoint materialization loads the complete checkpoint and patched state into
  host memory before writing shards; no constant-memory claim is made.
- The complete external Hugging Face patch workflow has not been established by
  the bundled preflight, which only loads/fingerprints a supplied local model and
  generates outputs. No third-party weights are committed.
- The general GPU path is not supported by committed execution evidence until a
  recorded GPU run exists.
- Stack and rebase evidence records do not yet have complete dedicated hostile-
  data parsers; the exact parser boundaries are documented in `SPEC.md`.
