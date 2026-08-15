# Compiler

The automatic compiler scores linear weights using aggregate target and guard
gradients, builds a contrastive low-rank basis, and optimizes explicit
primal-dual constraints. It tracks the best feasible candidate; it does not
silently convert a failed hard constraint into a weighted preference. Separate
APIs coordinate callback-driven CEGIS and executed individual-module/rank
pruning. The generic `compile` CLI performs deterministic bounded mutations of
declared non-holdout target and guard rows, executes every candidate, minimizes
supported failures, and recompiles exact/free-generation and base-KL
counterexamples. Unsupported assertion/search semantics return `UNSUPPORTED`
rather than being silently skipped. Sealed holdout paths remain inaccessible
until the final minimized candidate.

Semantic merge and rebase use a multi-contract variant. It also executes
`generation_length` as a discrete search assertion without presenting length as
a differentiable loss, balances bounded proposals across the contract union,
and retains parent or source-teacher objectives during recompilation. Sealed
holdout resources never enter this search plan.

The delta IR can apply sparse matrices and vectors, but the R1 automatic
compiler emits low-rank linear-weight deltas. It does not yet synthesize sparse
residuals, optimize embeddings/vectors, reoptimize minimization candidates, or
store per-example gradient covariances.
