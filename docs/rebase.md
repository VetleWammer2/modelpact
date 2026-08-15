# Semantic rebase

Direct transplantation is attempted only for compatible schemas and earns a
claim only after target, guard, and new-base preservation checks pass. Otherwise
the library protocol can provide the old patched model as target teacher and the
new base as preservation teacher to a supplied recompiler. The `rebase` CLI
supplies a low-rank recompiler for compatible built-in tiny-to-tiny transfers,
including controlled cross-architecture tiny models. It verifies the candidate
on the target base, opens a configured sealed holdout only for the final
candidate, and packages a new patch with rebase lineage. Every distinct bundled
contract is executed. Before transfer, the source patch is independently run on
visible target and preservation probes; a failing patched teacher is rejected.
Original guards are evaluated relative to the new unpatched base, and an
optional guard-only `--new-base-policy` adds explicit new-base controls. Bounded
CEGIS is executed over supported visible assertions from every original
contract plus the new-base policy before the sealed holdout is opened. Search
records every deterministic proposal and observation, minimizes failures, and
recompiles the accumulated working set while retaining the old patched model as
target teacher and the new unpatched model as guard teacher. Bounded module
removal applies to all supported delta state; matrix deltas additionally undergo
rank reduction. Successful semantic recompilation therefore carries a bounded
minimality claim rather than `UNMINIMIZED`. Direct-transplant results preserve
their source representation and do not claim semantic minimization.

Custom and Hugging Face adapters can take the structurally compatible direct
path, but failed or structurally incompatible transfers require an explicit
trusted recompilation backend. ModelPactBench separately executes same-family
and controlled cross-architecture analytic cases. Cross-architecture transfer
never maps physical tensors. The exercised tiny integration uses a source patch
that changes generated output, a target base with a different output,
teacher-guided optimization, mutated free-generation probes, and target-base
guards; generation length alone is not used as its transfer evidence.
