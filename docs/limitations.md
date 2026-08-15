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
- Generic CLI CEGIS searches deterministic mutations of declared visible probe
  rows. It does not invent an open-ended semantic prompt generator, and
  unsupported assertion/search combinations return `UNSUPPORTED`.
- The CLI provides joint merge and semantic rebase compilation for the built-in
  tiny adapter. Custom and Hugging Face repair/recompile paths require an
  explicit trusted compiler integration. Successful compose/merge results are
  complete bundles; failed candidates retain evidence but are not applicable
  patches.
- Stack resolution emits a complete, applicable composite bundle and executes
  `subset_audit_budget` when nonzero. The built-in tiny adapter can semantically
  repair a failed stack; repair for other adapters is explicitly unsupported.
- Rebase accepts a guard-only `--new-base-policy`, executes all distinct bundled
  contracts, and rejects a source patched teacher that fails visible contracts.
  Its bounded minimizer currently handles matrix-only semantic deltas; vector or
  mixed-state results remain honestly `UNMINIMIZED`.
- Greedy minimization tests individual module removals and rank truncations; it
  does not test module groups, reoptimize candidates, or prune sparse residuals.
- Checkpoint materialization streams one metadata-planned output shard at a time.
  It still retains patch factors and one target delta, and SafeTensors snapshots
  the current output shard while writing. A tensor larger than the shard limit is
  necessarily resident as one oversized shard. Reported peak RSS is a
  process-lifetime high-water mark where the platform exposes one, not an
  operation-isolated measurement; unsupported platforms record it as unavailable.
- The full offline Hugging Face workflow is established only for the generated
  one-layer GPT-NeoX fixture. Operator-supplied checkpoints currently receive a
  local-only load/fingerprint/generation preflight; this is not a compatibility
  claim for arbitrary Hugging Face model families. No third-party weights are
  committed.
- The general GPU path is not supported by committed execution evidence until a
  recorded GPU run exists.
- Rebase evidence records do not yet have a complete dedicated hostile-data
  parser; the exact parser boundary is documented in `SPEC.md`. Patch Stack
  Lockfile v1 has a strict core and CLI-extension parser.
