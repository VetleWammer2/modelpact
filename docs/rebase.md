# Semantic rebase

Direct transplantation is attempted only for compatible schemas and earns a
claim only after target, guard, and new-base preservation checks pass. Otherwise
the library protocol can provide the old patched model as target teacher and the
new base as preservation teacher to a supplied recompiler. The `rebase` CLI
supplies a low-rank recompiler for compatible built-in tiny-to-tiny transfers,
including controlled cross-architecture tiny models. It verifies the candidate
on the target base, opens a configured sealed holdout only for the final
candidate, and packages a new patch with rebase lineage. This CLI path currently
requires one deduplicated contract with preservation assertions.

Custom and Hugging Face adapters can take the structurally compatible direct
path, but failed or structurally incompatible transfers require an explicit
trusted recompilation backend. ModelPactBench separately executes same-family
and controlled cross-architecture analytic cases. Cross-architecture transfer
never maps physical tensors.
