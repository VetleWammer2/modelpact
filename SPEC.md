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

Schema v1 rejects unknown fields. A reader encountering a higher
`schema_version` returns `UNSUPPORTED`; it never silently interprets it as v1.
Artifact paths are bundle-relative POSIX paths without `.` or `..`, drive names,
absolute roots, NULs, or symlink traversal. Implementations impose bounded file
sizes, tensor dimensions, expression depth, collection counts, and integers
before allocation. Patch data cannot contain Python, callables, pickle, or an
expression that is evaluated as code.

Stable identities exclude timestamps, output directories, wall-clock timings,
hostnames, and report prose. Evidence can record those fields without changing
the underlying patch identity.

## Model Manifest v1

Required top-level fields are:

```text
schema_version = 1
adapter_id
architecture_hash
state_schema_hash
checkpoint_hash
tokenizer_hash
chat_template_hash
generation_config_hash
model_signature
state_schema
patchable_modules
non_patchable_persistent_state
aliases
parameter_count
dtype_distribution
runtime_modes
```

`checkpoint_hash` covers a sorted sequence of tensor key, shape, dtype, and
streaming content digest—not a directory name. `architecture_hash` covers the
canonical model configuration and adapter architecture identity.
`state_schema_hash` covers every persistent state key and all aliases.
Tokenizer identity covers tokenizer files and configuration. Chat-template and
generation-default identities are separate so a deployment-policy change cannot
hide inside checkpoint identity.

An alias group has one canonical state key and a sorted nonempty member set. A
manifest is invalid if groups overlap, a member is absent from state, shapes or
dtypes differ, or storage identity no longer agrees with the declared alias.
Cached fingerprints MAY be reused only after cache inputs—including sizes,
modification metadata, and canonical file inventory—are revalidated.

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

Every objective and assertion has a unique ID, safe relative source, type, and
type-specific bounded options. Numeric acceptance thresholds must be finite and
semantically valid. Unknown objective, assertion, scorer, or option fields are
rejected. A discrete assertion such as `json_schema` cannot be relabeled as a
differentiable objective; its compile proxy is separately declared.

Holdout sources cannot coincide with any compile, search, validation, or guard
source. A sealed source can be opened only through a stateful authorization token
for `final_candidate_only` or `independent_verification`. Opening it consumes the
token and writes an access record. A holdout failure terminates that contract
attempt; further optimization requires a new contract version and policy record.

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

Low-rank factors have shapes `[out, rank]` and `[rank, in]`, positive bounded
rank, identical floating dtype, and finite scale. Sparse indices are a strictly
sorted unique integer `[nnz, 2]` tensor in bounds; values are a floating `[nnz]`
tensor. Vector deltas are rank one. Alias nodes refer to another program target,
not a file. Cycles and partial updates to a tied alias group are rejected. Sum
terms must have the same shape and dtype. Unknown operations are rejected.

R1 semantics are exactly additive: applying target expression `e` to state
tensor `W` produces `W + materialize(e)`. Program order has no behavioral
semantics.

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

Data files listed in `manifest.json.artifact_hashes` are hashed before parsing.
Unlisted files do not contribute to patch identity and cannot be used to satisfy
a claim. Generated scripts are conveniences and MUST NOT be executed while
parsing or verifying a bundle.

The stable patch identity is the SHA-256 of length-delimited canonical manifest
identity fields, delta-program hash, sorted SafeTensors tensor digests, sorted
contract hashes, and base signature. The manifest stores no self-referential
`patch_id` during preimage calculation; after calculation the ID is inserted and
validated by recomputing the same preimage. Parent, merge, rebase, dependency,
and source-diff lineage are sorted where order is not semantic.

Mounting validates the base signature, module schema, alias groups, tensor keys,
shapes, and dtypes. Unknown operations or state are refused. Mounting twice is
either rejected or returns the existing identical mount; it cannot apply twice.
Unmount removes parameterizations and leaves the untouched base tensors. Folding
always targets a new output path, uses temporary same-filesystem files and atomic
rename, preserves non-weight configuration files, and emits a materialization
manifest.

## Patch Stack Lockfile v1

The canonical JSON lockfile pins:

```text
schema_version
base_signature and exact checkpoint_hash
sorted requested patch IDs and bundle hashes
contract hashes
dependency edges
resolution policy hash
resolution outcome
resolved artifact hash, when present
verification policy hash
certificate hash
composition-audit hash
```

The user-facing TOML input is declarative. Listed order does not alter additive
weight semantics. Dependency edges are checked for cycles. Outcomes are limited
to naïve verified stack, verified composite, partially resolved, static
contradiction, empirical failure, or unsupported. A lockfile never fetches a
missing patch or model.

## Verification Certificate v1

A certificate is regenerated by execution; bundled results are never trusted as
input evidence. Required fields include tool and environment identity, patch and
base identities, adapter, tokenizer, contract and probe hashes, generation
policy and seeds, objectives, every target and guard result, holdout and free
generation results, prompt-level metrics, intervals, counterexample budget and
findings, active modules/ranks/sparsity/bytes, minimization, composition,
interaction and rebase results, artifact hashes, warnings, and unsupported
claims.

Outcomes are exactly `PASS`, `FAIL`, `INCONCLUSIVE`, `UNSUPPORTED`, and
`NOT_APPLICABLE`. Only `PASS` satisfies a required assertion. Prompt text MAY be
redacted, but its hash and output hash are mandatory. A certificate claim must
be in the claim taxonomy and pass the corresponding structural and execution
predicate. `GLOBAL_MINIMUM` additionally requires an exhaustive relevant search
record.

Normative wording is:

> Verified under the declared contracts, probe spaces, generation policy,
> environment, and search budget.

## Composition Audit v1

The audit records patch IDs, possible subset count, executed inclusion vectors,
per-contract signed margins, execution order, model/base identity, initial design,
surrogate configuration, fitted terms, uncertainty method, active-selection
trace, failing subsets, and ddmin traces.

`EXHAUSTIVE_COMPOSITION_AUDIT` requires every nonempty subset plus the empty
stack to have executed. Otherwise the only success-like negative finding is
`NO_FAILURE_FOUND_WITHIN_BUDGET`, paired with `BUDGETED_COMPOSITION_AUDIT` and
the exact execution budget. Surrogate predictions cannot populate executed
outcomes.

## Rebase Evidence v1

Rebase evidence pins source patch/base and target base identities, semantic
compatibility checks, the direct-transplant artifact and executed result, any
recompile objectives and budgets, old-patched teacher evidence, new-base guard
evidence, complexity change, CEGIS/minimization traces, holdout result, and final
claim.

`DIRECT_TRANSPLANT_VERIFIED` requires an actually applied delta and passing old
targets, old guards, and new-base preservation checks. Different architectures
cannot use tensor transplantation. `SEMANTIC_REBASE_VERIFIED` requires a newly
compiled patch and passing target/new-base contracts. Optimization failure is
`REBASE_FAILED` or `REBASE_INCONCLUSIVE`, never proof of impossibility.

## Security limits

The reference implementation defaults to a maximum JSON/YAML depth of 64, delta
expression depth 32, 4,096 sum terms, 100,000 program targets, 1 MiB JSONL record,
64 MiB probe file, 1,000,000 probes, and explicit maximum tensor/file byte
budgets. Implementations MAY choose lower limits but cannot silently choose
higher ones when reading untrusted data. YAML aliases, custom tags, duplicate
keys, and recursive objects are rejected. SafeTensors metadata is data only and
is bounded before tensor allocation where the library permits.

