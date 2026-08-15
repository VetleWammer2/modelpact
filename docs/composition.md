# Composition and semantic merge

Composition sums additive deltas and then executes the union of every target and
guard contract. Static overlap and subspace metrics are diagnostics only. A
failed composition is not repaired by `compose`.

The semantic-merge library protocol passes parent deltas, union contract IDs,
the summed initializer, identities, and a budget to a supplied joint compiler,
then independently executes the returned candidate. ModelPactBench includes a
concrete analytic optimizer. The `merge` CLI supplies a real constrained
low-rank joint compiler for the built-in tiny adapter, using declared objectives,
parent-teacher probes, and union guards. Recompiled tiny candidates then enter a
bounded CEGIS refinement. Deterministic mutations are balanced across supported
visible target and guard assertions in the union; every proposal is executed,
failures are minimized, and accumulated counterexamples are supplied to another
joint optimization. Executed module-removal and matrix-rank candidates follow.
The evidence records proposals, observations, compilation candidates, rounds,
counterexamples, and minimization candidates. A successful tiny semantic merge
does not emit `UNMINIMIZED`; its minimality claim is bounded to those executions.

Successful `compose` and `merge`
outputs are complete Behavior Patch Bundle v1 directories with union contracts,
copied probe/schema/holdout resources, independent certificates, interaction
evidence, and standalone tools. Search phases cannot inspect sealed holdout;
each configured holdout is executed once for the selected final candidate.
Custom and Hugging Face adapters require an explicit trusted compiler
integration when repair is needed; that case is reported `UNSUPPORTED` and must
not be described as repaired.

Declarative stack resolution uses the same executed closure and packaging path.
A nonzero `subset_audit_budget` runs actual subset candidates before the final
selection, and the lockfile pins the resulting audit, certificate, and resolved
bundle manifest hashes. `repair_conflicts = true` invokes the tiny semantic
merge backend only after naïve closure fails and uses the same CEGIS and
minimization path as `modelpact merge`; unsupported adapter backends fail
honestly instead of emitting an unusable raw delta.
