# ModelPact R1 Artifact Specification

Status: working R1 specification, schema version 1. Normative words `MUST`, `MUST
NOT`, `SHOULD`, and `MAY` are used in their usual standards sense.

## Common rules

All JSON artifacts use UTF-8 and the ModelPact v1 canonical encoding: objects are
sorted by Unicode key, insignificant whitespace is removed, non-finite numbers
are rejected, negative zero is normalized to zero, and bytes are never embedded
as JSON strings. Hashes are lowercase `sha256:<64 hexadecimal digits>`. YAML is
an authoring format only; identity is computed after parsing into the normalized
JSON data model.

Unknown-field policy is artifact-specific. Behavior Contract, Delta Program,
Patch Manifest, Verification Certificate, Rebase Evidence, and Patch Stack
Lockfile readers reject unknown fields in their closed schemas. The current
Model Manifest/State Schema readers validate the fields they use but tolerate
additional fields. A reader never interprets an unknown `schema_version` as v1;
command surfaces report unsupported versions as an error/non-success state.

The Rebase Evidence and on-disk Patch Stack Lockfile readers require the input
bytes to be exactly the ModelPact canonical JSON encoding, optionally followed
by the single LF emitted by ModelPact writers. They do not silently normalize a
BOM, CRLF, insignificant whitespace, alternate escapes, key order, number
spelling, or trailing content. Duplicate keys and non-finite numbers are rejected
before schema construction.

Paths embedded in bundles and verification resources are bundle-relative POSIX
paths in one exact portable spelling. Readers reject backslashes; empty, `.` or
`..` components; drive, UNC, or absolute roots; NUL/control characters; Windows
reserved device-name stems; alternate-data-stream colons and other characters
forbidden in Windows names; and components ending in a dot or space. A path set
MUST NOT contain spellings that collide after Unicode case folding, and fixed
reserved artifact paths MUST use their specified case. Resolution MUST NOT
traverse a symlink. A user-selected CLI output path is not an embedded artifact
path, although local paths serialized in the CLI stack-lock extension are also
required to use canonical, portable POSIX syntax.

Readers impose artifact-specific bounds on file sizes, tensor dimensions,
expression depth, collection counts, and integers before the corresponding
allocation. Patch data cannot contain Python, callables, pickle, or an
expression that is evaluated as code.

Stable identities exclude timestamps, output directories, wall-clock timings,
hostnames, and report prose. Evidence can record those fields without changing
the underlying patch identity.

## Model Manifest v1

The reference writer emits these top-level fields:

```text
schema_version = 1
signature
  schema_version = 1
  adapter_id
  architecture_hash
  state_schema_hash
  checkpoint_hash
  tokenizer_hash
  chat_template_hash
  generation_config_hash
state_schema
checkpoint_tensor_hashes
parameter_count
patchable_parameter_count
dtype_distribution
supported_runtime_modes
unsupported_state
metadata
```

`signature` is the portable `ModelSignature` object. `checkpoint_hash` is the
canonical hash of the sorted key-to-tensor-content-hash mapping; each tensor
content hash covers dtype, shape, and exact contiguous CPU bytes. It is not a
directory-name identity. `architecture_hash` covers adapter ID, the canonical
model configuration, and emitted module specifications. `state_schema_hash`
covers the emitted tensor/module/alias schema. Tokenizer identity covers the
present files in the v1 allowlist (`tokenizer.json`, tokenizer configuration,
special/added-token files, vocab/merges, and SentencePiece/tokenizer model
files). Chat-template and generation-default identities are separate.

An alias group has one canonical state key and a sorted nonempty member set. A
group contains at least two members and uses its lexicographically first member
as canonical. State-schema parsing checks that every member is a known tensor;
runtime alias-map construction additionally rejects overlapping groups. Manifests
built from a live model derive aliases from shared storage identity, but parsing
a serialized manifest does not independently re-establish storage identity or
compare alias-member shapes/dtypes. R1 has no fingerprint-cache implementation.

The reader requires the signature, state schema, checkpoint tensor hashes,
counts, and dtype distribution. It accepts omitted runtime/unsupported/metadata
fields with empty defaults and currently ignores unknown fields. Consumers MUST
not assign semantics to such unknown fields.

## Behavior Contract v1

The normalized shape is:

```yaml
schema_version: 1
id: bounded-stable-id
contract_version: 1
model_requirements: {}
compile:
  objectives: []
verify:
  targets: []
  guards: []
holdout: {}
statistics: {}
generation: {}
```

Compilation objectives and verification assertions are disjoint AST node types.
The required objective vocabulary is:

```text
teacher_cross_entropy  teacher_kl  preferred_sequence_margin
base_kl                hidden_state_matching  activation_direction
```

The required assertion vocabulary is:

```text
token_log_probability  sequence_log_probability  sequence_margin
multiple_choice_margin exact_match               normalized_exact_match
regular_expression     json_parse                json_schema
free_generation_match  reference_kl              base_kl
generation_length      perplexity
```

Objective IDs are unique within the compile list; assertion IDs are unique
across targets and guards (an objective and assertion may share text). Every
node has a bounded source string, a type, and type-specific bounded options.
Source confinement is enforced when the
resource is opened through `resolve_contract_resource` or the model-backed
provider; constructing the AST alone does not prove that the string is a safe
path. Numeric acceptance thresholds must be finite and semantically valid.
Unknown objective, assertion, scorer, or option fields are rejected. A discrete
assertion such as `json_schema` cannot be relabeled as a differentiable
objective; its compile proxy is separately declared.

Holdout sources cannot be identical to an objective, target-assertion, or
guard-assertion source. A sealed source can be opened only through a stateful
authorization capability for `final_candidate_only` or explicitly enabled
`independent_verification`. Authorization consumes one gate; each validated role
access writes an in-memory access record. The gate cannot be reused after a
holdout failure, but enforcement of the research rule that a new attempt use a
new contract version/policy is the responsibility of orchestration and retained
run records, not a global cryptographic mechanism.

The contract hash is SHA-256 of its complete canonical normalized form.

## Delta Program v1

The program is a safe additive AST:

```json
{
  "schema_version": 1,
  "targets": {
    "layers.0.mlp.up_proj.weight": {
      "op": "sum",
      "terms": [
        {"op": "low_rank_matrix", "left": "l0.B", "right": "l0.A", "scale": 1.0},
        {"op": "sparse_matrix", "indices": "l0.i", "values": "l0.v", "shape": [64, 32], "scale": 1.0}
      ]
    }
  }
}
```

Operations are `low_rank_matrix`, `sparse_matrix`, `vector`, `alias`, and
`sum`. Each operation supports shape inference, validation, materialization,
application, referenced-byte estimation, and canonical serialization.

Low-rank factors have shapes `[out, rank]` and `[rank, in]`, positive rank,
identical floating dtype, and finite scale; file/tensor limits bound their
allocation. Sparse indices are a strictly sorted unique integer `[nnz, 2]`
tensor in bounds; values are a floating `[nnz]` tensor. Vector delta tensors are
rank one. Alias nodes refer to another program target, not a file, and alias
cycles are rejected. When a model state schema is supplied, partial or
inconsistent updates to a tied alias group are rejected. Sum terms must have the
same shape and dtype. Unknown top-level or operation fields and unknown
operations are rejected.

R1 semantics are exactly additive: applying target expression `e` to state
tensor `W` produces `W + materialize(e)`. Target-map order has no semantic
meaning. A `sum` term list is serialized and evaluated in its listed order;
floating-point reassociation is not claimed to be bitwise invariant.

## Behavior Patch Bundle v1

The directory layout is:

```text
manifest.json
delta-program.json
tensors.safetensors
contracts/
probes/
evidence/
certificate.json
report.md
apply_patch.py
verify_patch.py
```

Data files listed in `manifest.json.artifact_hashes` are hashed before the delta
program and tensors are parsed. Unlisted files do not contribute to patch
identity and cannot be used to satisfy a claim. Generated scripts are
conveniences and MUST NOT be executed while parsing or verifying a bundle. The
low-level constructor/loader can operate on a core bundle; callers that require
the complete R1 layout must call the complete-bundle check. Release-facing
compile/extract/rebase packaging paths request that complete check.

ModelPact exposes three related, deliberately distinct content addresses:

* `patch_id = hash_canonical(identity_payload)` identifies the executable
  behavior delta, its base/schema compatibility, executable contracts, claims,
  lineage, and compiler configuration.
* `evidence_id = hash_canonical(evidence_payload)` binds that `patch_id` to all
  hashed probes, evidence, reports, deltas, tensors, and contracts. The bundled
  certificate and generated helpers are excluded because they embed these
  identities and would otherwise create a hash cycle. Generated helpers pin and
  re-check this ID before applying or verifying a patch.
* `bundle_id = hash_canonical(manifest)` addresses the complete manifest after
  certificate and generated helpers have been attached. Stack lockfiles pin the
  manifest hash as their immutable bundle reference.

The stable core patch identity payload is the manifest payload without
`patch_id`, with `artifact_hashes` filtered to `delta-program.json`,
`tensors.safetensors`, and files below `contracts/`. Thus it includes the base
signature, schema, tool version, name, delta representation, target schema,
contract/dependency/lineage identifiers, compiler configuration,
verification-policy hash, and the exact file hashes of the program,
SafeTensors container, and contracts. It does not separately hash per-tensor
digests. Evidence is post-core-ID so certificates can name the patch, but it is
not mutable or unaudited: `evidence_id` binds it, and standalone tools reject
evidence mutation even when an attacker recomputes the manifest artifact hash.
After calculation the core ID is inserted and validated by recomputing the same
canonical payload. Parent, merge, and dependency lists are sorted where order
is not semantic.

A bundle whose manifest has `rebased_from` MUST list both the bare canonical
`evidence/rebase.json` record and a canonical
`evidence/source-manifest.json`. The latter is a complete, self-validating Patch
Manifest v1 for the source patch. Its `patch_id`, full base signature, and
`provides` set bind the Rebase Evidence source patch, source base, and old
target-contract identities. A non-rebased bundle MUST NOT carry either reserved
rebase-lineage artifact.

Bundle mounting validates the base signature, module schema, alias groups,
tensor keys, shapes, and dtypes. Unknown operations or state are refused.
Mounting a second ModelPact patch on an already mounted model is rejected.
Unmount removes parameterizations and leaves the untouched base tensors.
Folding always targets a new output path, uses temporary same-filesystem files
and directory rename, copies regular top-level non-weight files, and emits a
materialization manifest. The writer validates SafeTensors headers, the logical
state/alias schema, and the complete delta program before it creates an output
directory. It then plans deterministic shards from tensor metadata and loads,
patches, hashes, and writes one planned output shard at a time. Physically omitted
tied keys are expanded from a verified stored member; multiple stored copies must
have identical content. Unsupported alias layouts are rejected.

This is bounded streaming, not constant-memory materialization. Peak working
state includes patch factors, one materialized target delta, one planned output
shard, and the SafeTensors writer's snapshot copy. A single tensor larger than
`max_shard_size` necessarily forms an oversized one-tensor shard. The manifest's
`performance` object records the configured bound, largest tensor and planned
shard payloads, tensor/auxiliary read bytes and elapsed time, checkpoint write
bytes and elapsed time, and the process-lifetime peak RSS high-water mark. RSS is
measured with Linux `VmHWM` or Unix `getrusage` where supported; otherwise the
value is null and the method is explicitly `unavailable`. Source hash passes and
the manifest's own write are excluded from the read/write counters as declared
by `measurement_scope`.

## Patch Stack Lockfile v1

The core `StackLock.to_dict()` record contains exactly:

```text
schema_version = 1
base_hash
patch_hashes                 # patch ID -> patch manifest-file hash
contract_hashes
resolved_artifact_hash       # nullable
verification_policy_hash
resolution
certificate_hash             # nullable
audit_hash                   # nullable
```

`base_hash` is a caller-supplied base identity; the CLI uses the exact checkpoint
tensor fingerprint rather than a complete nested `ModelSignature`.
`patch_hashes` is likewise supplied by `PatchReference`; the CLI maps each patch
ID to its manifest-file hash. Patch IDs are emitted in sorted mapping order and
contract hashes are sorted/deduplicated. The current core lock does not serialize
dependency edges, requested order, repair policy, or a separate resolution-policy
hash. The CLI adds an `extensions.modelpact_cli` object containing local paths, a
base-manifest hash, dependency order, and the complete resolved-patch path; that
extension is outside the core dataclass. Successful resolution pins the resolved
bundle manifest hash, regenerated certificate hash, and—when requested—the
executed composition-audit file hash.

The user-facing TOML input is declarative. Listed order does not alter additive
weight semantics. The library topological sorter interprets `requires` and
`provides` as contract hashes, resolves each requirement to an unambiguous
provider patch, and rejects missing requirements and cycles. The CLI performs
the same validation before composition or output creation. A nonzero
`subset_audit_budget` executes real visible-probe subset evaluations; it never
opens sealed holdouts, which remain reserved for the selected final candidate.
Outcomes are
`NAIVE_ADDITIVE_STACK`,
`VERIFIED_COMPOSITE_PATCH`, `PARTIALLY_RESOLVED_STACK`,
`STATIC_CONTRADICTION`, `EMPIRICAL_FAILURE`, or `UNSUPPORTED`. A lockfile never
fetches a missing patch or model.

`StackLock.from_dict` rejects unknown or missing core fields, malformed hashes,
unsupported resolution values, duplicate/unsorted contract identities, and
oversized patch or contract collections. The CLI envelope admits only the
documented `modelpact_cli` extension, bounds absolute local paths, requires path
and dependency maps to cover the locked patch set exactly, and is parsed only
from bounded canonical JSON. Before the base model or any patch is loaded, the
referenced-manifest preflight requires regular paths with no symlink traversal,
enforces per-file and aggregate byte limits, authenticates each manifest-file
hash, strictly parses and validates the manifest, and requires its `patch_id` to
equal the corresponding `patch_hashes` key. Before model loading, the same
preflight parses and authenticates any canonical Rebase Evidence and companion
source-manifest records carried by each input or resolved patch, along with all
of their executable target/guard contracts. It validates contract roles against
`provides`/`preserves`,
checks the resolved certificate and artifact projection, checks the locked full
base and contract identities against the referenced manifests, and validates
the recorded dependency order against their declared requirements and
providers. Input and resolved manifests together may reference at most 100,000
artifacts. A recomputed lockfile cannot substitute a different valid manifest
under an old patch identity. A data-only core lock without path metadata remains
useful for content exchange, but a consumer cannot claim external-artifact
validation without caller-supplied references or expectations.

Revert evidence uses these grades:

```text
RUNTIME_UNMOUNT_EXACT
BASE_HASH_RESTORED
NUMERIC_DELTA_INVERSE
VERIFIED_LOGICAL_STACK_RECONSTRUCTED
SEMANTIC_STACK_RECOMPILED
REVERT_FAILED
```

`VERIFIED_LOGICAL_STACK_RECONSTRUCTED` means the untouched base plus the
remaining original patches were resolved and their contracts re-executed.
`SEMANTIC_STACK_RECOMPILED` is reserved for a new optimization run. Neither
grade implies bitwise inversion of a folded floating-point delta.

## Verification Certificate v1

A certificate built from a verification report contains tool and environment
identity, patch and base identities, adapter, checkpoint/tokenizer/contract/probe
hashes, verification and generation policy, seeds, compile-objective
descriptions, target/guard/holdout and free-generation results, prompt-level
metrics, intervals, counterexample, patch-structure, minimization, composition,
interaction and rebase records, artifact hashes, warnings, unsupported claims,
compatibility errors, overall outcome, the verification-result hash, and its own
content hash. Some subsystem records may explicitly say `NOT_EXECUTED`,
`UNMINIMIZED`, or `NOT_APPLICABLE`; their presence is not evidence that the
subsystem ran.

Every serialized assertion result includes an `acceptance_policy` projection of
the exact executable threshold used for that result. Binary assertions record
`minimum_pass_rate`; continuous assertions record the applicable minimum,
maximum, mean, item, or quantile limits. The reader recomputes aggregate values,
prompt outcomes, and margins from that projection. This lets a legitimate
aggregate pass contain permitted prompt-level failures while preventing a
re-hashed certificate from relabeling an out-of-policy continuous value as PASS.

Parsing a bundled certificate validates its strict field set, self-hash, digest
syntax, known claim names, and selected claim/evidence consistency. A concrete
Rebase Evidence record embedded in `rebase_result` is parsed by the same Rebase
Evidence v1 rules and its repeated source patch, source base, target base, and
claim fields must agree. Its artifact set MUST include both
`evidence/rebase.json` and `evidence/source-manifest.json`; `NOT_APPLICABLE`
remains an explicit distinct shape. When an artifact root is supplied,
validation also strictly parses the canonical target `manifest.json`, both
lineage artifacts, and the executable contracts. The target manifest and parsed
contract documents bind `new_patched_behavior` to target contracts and
`new_base_preservation` to guard contracts, rather than accepting their union as
an interchangeable set. `rebase_result` records the packaging-time rebase
lineage, not the outcome of this execution, so a re-verified rebased bundle that
now fails its contracts still emits a `FAIL` certificate carrying that lineage.
`RebaseClaim` values are not admissible in `claims`; a rebase claim is asserted
only through a strictly parsed `rebase_result`. The
`patch_id` and flattened `base_signature`, like every other certificate digest,
MUST be exactly `sha256:` followed by 64 lowercase hexadecimal digits. It does
not re-execute a model. `independently_verify` instead hashes a caller-declared
artifact set, executes the supplied model-backed provider, builds a new
certificate, and uses any prior certificate only for comparison.

Outcomes are exactly `PASS`, `FAIL`, `INCONCLUSIVE`, `UNSUPPORTED`, and
`NOT_APPLICABLE`. Only `PASS` satisfies a required assertion. Prompt text is not
stored in prompt metrics; its hash is mandatory. An output hash is present when
generated text exists, and free-generation records always carry prompt, output,
token-ID, and generation-policy hashes.

Certificate parsing rejects unknown claim names and directly checks evidence for
base compatibility, target assertions, preservation assertions, sealed holdout,
and the presence of free-generation records. Objective-optimized and
minimized-within-budget claims are emitted by the builder from explicit caller
booleans; their nested evidence is not yet fully re-derived by the parser.
`GLOBAL_MINIMUM` is in the shared taxonomy but is not emitted by the builder and
MUST NOT be asserted by this implementation.

Normative wording is:

> Verified under the declared contracts, probe spaces, generation policy,
> environment, and search budget.

## Composition Audit v1

The implemented `AuditResult` records patch IDs, possible nonempty-subset count,
the nonempty subset budget, an optional empty-stack baseline, ordered executed
subset evaluations with per-contract signed margins/outcomes, claims, coverage,
failing subsets, ddmin results/tested candidates, active-proposal scores,
surrogate-fit summaries, and search-space/budget exhaustion flags. Fit summaries
contain contract ID, observation count, selected regularization strength,
cross-validation MSE, and degree; fitted coefficient maps and full bootstrap
members are not retained. The CLI wraps this record in `audit.json` and hashes it
from a small manifest, but there is no dedicated hostile-data Audit v1 parser or
self-contained model/base identity field yet.

`EXHAUSTIVE_COMPOSITION_AUDIT` currently requires every nonempty subset to have
executed; the empty-stack baseline is enabled by default and recorded separately,
but is not part of the code predicate for that claim. Otherwise the only
success-like negative finding is `NO_FAILURE_FOUND_WITHIN_BUDGET`, paired with
`BUDGETED_COMPOSITION_AUDIT` and the exact nonempty-subset budget. A budgeted run
that observes failure instead records `FAILING_SUBSET_FOUND`. Surrogate
predictions cannot populate executed outcomes.

## Rebase Evidence v1

The serializable `RebaseEvidence.to_dict()` record contains schema version,
source patch ID, source/target base hashes, claim, compatibility classification,
whether direct transfer was attempted and its outcome, whether recompilation was
attempted, executed step/restart counts, budget exhaustion, old-patched behavior
margins, new-patched behavior margins, new-base preservation margins, before/
after complexity summaries, warnings, and `evidence_hash`. The evidence hash is
the lowercase tagged SHA-256 hash of the complete canonical record payload with
`evidence_hash` omitted. It detects an un-rehashed mutation but, like every R1
hash, does not authenticate a publisher.

The Rebase Evidence v1 reader accepts only bounded canonical JSON with the exact
closed field set. It rejects duplicate keys, missing or unknown fields, unknown
schema/claim/compatibility/outcome values, malformed identities, a stale
`evidence_hash`, non-finite metrics, negative or out-of-range counts and
complexities, oversized strings or collections, and inconsistent attempt,
outcome, budget, and claim state. In particular,
`DIRECT_TRANSPLANT_VERIFIED` requires an attempted passing direct transfer on a
directly compatible physical schema and no recompile; a semantic verified claim
requires executed recompilation and passing finite target/preservation margins.
No-recompile records have zero recompile counts and cannot report budget
exhaustion. Recomputing `evidence_hash` after changing those fields does not make
an inconsistent claim valid.

Callers may pin the evidence hash, source patch ID, source base hash, target base
hash, claim, source target-contract identity set, target-contract identity set,
and preservation-contract identity set as external expectations. Expectation
validation happens in addition to the self-hash and semantic checks. A
successful rebased Behavior Patch Bundle MUST carry both canonical
`evidence/rebase.json` and canonical `evidence/source-manifest.json`. The source
manifest MUST self-validate, its `patch_id` MUST equal the target manifest's
`rebased_from`, and its full base signature and combined `provides`/`preserves`
sets bind `source_base_hash` and `old_patched_behavior`. A semantic rebase
re-measures the source patched model on every contract the source bundle
carries, including preservation-only ones. The target manifest's full base
signature, `provides`, and `preserves` bind `target_base_hash`,
`new_patched_behavior`, and `new_base_preservation`. Repeated certificate rebase
fields and its nested evidence must agree with the artifact record as well. With
an artifact root, certificate validation additionally parses the target
manifest and executable contracts so target and guard roles cannot be swapped
while retaining the same identity union. The record still does not embed the
candidate artifact, contract documents, CEGIS/minimization traces, or a holdout
record; those are bound by the containing bundle manifest and verification
certificate.

On unsuccessful CLI orchestration, a file named `rebase-evidence.json` is a
diagnostic command report and may wrap a bare Rebase Evidence record together
with failure, CEGIS, or verification details. It is not itself Rebase Evidence
v1 and MUST NOT be passed to the bare-record reader or treated as a verified
claim. Successful rebased bundles carry the bare normative record at
`evidence/rebase.json`.

`DIRECT_TRANSPLANT_VERIFIED` requires an actually applied delta and passing old
targets, old guards, and new-base preservation checks. Different architectures
cannot use tensor transplantation. `SEMANTIC_REBASE_VERIFIED` requires a newly
compiled patch and passing target/new-base contracts. Optimization failure is
`REBASE_FAILED` or `REBASE_INCONCLUSIVE`, never proof of impossibility.

These semantics are enforced by the library orchestration when supplied with
applier, verifier, teacher-builder, and behavioral-recompiler callbacks. The CLI
implements and packages both the direct verified path and a low-rank behavioral
recompiler for built-in tiny-to-tiny transfers. That tiny recompiler uses the
old patched model on declared objective probes, applies the declared target
objectives and new-base guards, and executes final verification plus the sealed
holdout when configured. It currently requires one deduplicated guarded contract.
Custom and Hugging Face adapters can use direct verification, but a failed or
structurally incompatible transfer needs an explicit trusted recompilation
backend and is reported as unsupported/inconclusive rather than verified.

## Security limits

Limits are reader-specific. Canonical JSON serialization defaults to depth 64.
Behavior Contract parsing defaults to 2 MiB, depth 32, 100,000 nodes, 10,000
object keys, 10,000 objectives, and 50,000 assertions. Verification probe loading
defaults to a 64 MiB file, 2 MiB line, and 100,000 records; the separate generic
probe-dataset loader permits 1 MiB lines and up to 1,000,000 probes. Compilation
JSONL permits a 64 MiB source, 2 MiB record, and 1,000,000 records.

Delta programs permit expression depth 32, 4,096 sum terms, 100,000 targets,
and 2,048-character tensor names. Every target's inferred dense output is capped
at `2^30` elements and 4 GiB. Shape, dtype, schema, alias, and output-size checks
use metadata-only tensors before any low-rank multiplication or sparse
densification. SafeTensors loading defaults to 16 GiB per file, 100,000 tensors,
and `2^40` elements per tensor. Patch manifests are limited to 16 MiB and a
single supplemental-artifact attachment operation to 512 MiB. These are input
bounds, not memory-usage guarantees for an explicitly authorized application.

YAML anchors/aliases/explicit tags, duplicate YAML/JSON keys, non-finite numbers,
and recursive objects are rejected by the strict data parser. SafeTensors is
treated as data only. The loader validates regular-file status, total file size,
key count/name, and tensor element count, but the current implementation does
not separately pre-parse and bound every SafeTensors metadata field before
opening the container.

Rebase Evidence permits 16 MiB, depth 16, 500,000 total nodes,
4,096-character strings, 100,000 behavior references per behavior map, 1,024
complexity metrics per summary, and 10,000 warnings. A behavior or complexity
reference key is a nonempty string of at most 256 characters carrying no path
separator, control character, surrounding whitespace, or bare `.`/`..`
spelling. Contract identifiers are otherwise unconstrained, so the reader bounds
key shape and leaves identity binding to caller expectations. Executed
step/restart counts
are non-negative signed 32-bit integers; integral complexity values are
non-negative signed 64-bit integers. Patch Stack Lockfiles permit 16 MiB, depth
16, 150,000 total nodes, 10,000 keys per object, 4,096 patches, 100,000 contract
identities, 4,096-character embedded local paths, and 100,000 aggregate artifact
references across input and resolved patch manifests. Referenced manifests,
rebase/source records, executable contracts, and the resolved certificate are
preflighted under a 512 MiB aggregate limit in addition to their per-file limits.
These bounds are checked before higher-level rebase, resolution, revert, or
model-loading logic consumes the records.
