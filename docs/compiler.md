# Compiler

The compiler scores candidate modules using target and guard gradients, builds a
contrastive low-rank basis, optimizes explicit primal-dual constraints, searches
for counterexamples, and executes module/rank pruning. It tracks the best
feasible candidate; it does not silently convert a failed hard constraint into a
weighted preference. Sealed holdout data is outside the compiler API.

