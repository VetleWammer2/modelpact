# Concepts

A Behavior Patch is an additive parameter program plus executable contracts,
identities, lineage, and evidence. Target objectives state how it is trained;
target assertions state how the behavior is accepted; guards state what must
remain base-like. Sealed holdouts are opened only after candidate selection.

Adding patches is ordinary tensor addition. “Contract closed” means the combined
model still passes every parent target and guard. If it does not, semantic merge
can ask a supplied joint compiler to train a new delta against the union. Rebase
first verifies direct transfer and can ask a supplied behavioral compiler to
recompile on the new base. The CLI provides concrete low-rank joint compilation
and semantic rebase for the built-in tiny adapter; custom and Hugging Face
recompilation requires an explicit trusted backend. The shipped analytic
research drivers exercise the same orchestration boundaries. Every conclusion
is scoped to the executed contract, probe space, environment, and budget.
