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

