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

## Evidence boundary

The bare `RebaseEvidence` record is untrusted, content-addressed data. Its strict
reader requires canonical UTF-8 JSON (optionally followed by one LF), an exact
v1 field set, bounded shape and collections, lowercase tagged SHA-256 source and
base identities, finite metrics, bounded non-negative execution/complexity
counts, and a valid `evidence_hash` over the canonical payload without that hash
field. It also derives attempt/outcome/claim consistency, so an attacker cannot
turn a failure into `DIRECT_TRANSPLANT_VERIFIED` or
`SEMANTIC_REBASE_VERIFIED` merely by recomputing the content hash. Callers can
pin the evidence hash, source/target identities, claim, and source-target,
target, and preservation contract identity sets as additional expectations.

A successful rebased patch stores the bare record at `evidence/rebase.json` and
a complete canonical copy of its source Patch Manifest v1 at
`evidence/source-manifest.json`. Bundle loading self-validates that companion,
requires its `patch_id` to equal `rebased_from`, and derives the expected source
base hash and `old_patched_behavior` identities from its base signature and
`provides` set. The target manifest separately binds the target base,
`new_patched_behavior`, and `new_base_preservation`. Repeated certificate fields
and nested evidence must agree with the same artifact record. When certificate
validation has an artifact root, it also parses the canonical target manifest
and executable contract files and checks their target/guard roles, preventing a
rehash that merely swaps identities between the two evidence maps. Non-rebased
bundles cannot carry either reserved lineage artifact.

By contrast, `rebase-evidence.json` written beside an unsuccessful CLI run is a
diagnostic report that may wrap core evidence and other failure details; it is
not the bare v1 record and must not be interpreted as a verified claim.

Custom and Hugging Face adapters can take the structurally compatible direct
path, but failed or structurally incompatible transfers require an explicit
trusted recompilation backend. ModelPactBench separately executes same-family
and controlled cross-architecture analytic cases. Cross-architecture transfer
never maps physical tensors. The exercised tiny integration uses a source patch
that changes generated output, a target base with a different output,
teacher-guided optimization, mutated free-generation probes, and target-base
guards; generation length alone is not used as its transfer evidence.
