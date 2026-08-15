# Compiler

The automatic compiler scores linear weights using aggregate target and guard
gradients, builds a contrastive low-rank basis, and optimizes explicit
primal-dual constraints. It tracks the best feasible candidate; it does not
silently convert a failed hard constraint into a weighted preference. Separate
APIs coordinate callback-driven CEGIS and executed individual-module/rank
pruning. The generic `compile` CLI reports `UNSUPPORTED` for requested CEGIS
rounds unless an explicit search backend is available; ModelPactBench research
drivers provide bounded concrete loops. Sealed holdout paths are outside the
low-rank compiler API.

The delta IR can apply sparse matrices and vectors, but the R1 automatic
compiler emits low-rank linear-weight deltas. It does not yet synthesize sparse
residuals, optimize embeddings/vectors, reoptimize minimization candidates, or
store per-example gradient covariances.
