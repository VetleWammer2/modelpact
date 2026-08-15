# Concepts

A Behavior Patch is an additive parameter program plus executable contracts,
identities, lineage, and evidence. Target objectives state how it is trained;
target assertions state how the behavior is accepted; guards state what must
remain base-like. Sealed holdouts are opened only after candidate selection.

Adding patches is ordinary tensor addition. “Contract closed” means the combined
model still passes every parent target and guard. If it does not, semantic merge
trains a new patch against the union. Rebase first verifies direct transfer and,
if needed, recompiles behavior on the new base. Every conclusion is scoped to
the executed contract, probe space, environment, and budget.

