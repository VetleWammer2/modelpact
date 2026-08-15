# Composition and semantic merge

Composition sums additive deltas and then executes the union of every target and
guard contract. Static overlap and subspace metrics are diagnostics only. A
failed composition is not repaired by `compose`.

The semantic-merge library protocol passes parent deltas, union contract IDs,
the summed initializer, identities, and a budget to a supplied joint compiler,
then independently executes the returned candidate. ModelPactBench includes a
concrete analytic optimizer. The `merge` CLI supplies a real constrained
low-rank joint compiler for the built-in tiny adapter, using declared objectives,
parent-teacher probes, and union guards. Its output is an executed composition
artifact (resolved dense delta, union contracts, and verification evidence), not
a complete Behavior Patch Bundle v1. Custom and Hugging Face adapters require an
explicit trusted compiler integration when repair is needed; that case is
reported `UNSUPPORTED` and must not be described as repaired.
