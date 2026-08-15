# ModelPact

Compile, compose, rebase, audit, and revert learned behavior.

A deterministic tiny causal LM was trained with five new behaviors. ModelPact
found five scoped difference clusters, selected one, compiled it into a rank-2
patch, and rejected all four unselected changes on the declared controls. Target
transfer, unselected-change rejection, and unchanged-control preservation were
all 1.000 on the executed suites; sealed holdout and free generation passed.

Two independently valid patches later failed when added: the format prompt
generated `r` instead of `{`. A fresh 100-step semantic merge passed both parent
contracts. Direct transplant of the fact patch onto base-v2 failed; a 120-step
behavioral recompile restored the fact while retaining base-v2's new `F:b -> D`
behavior. These are finite experimental results, not universal guarantees.

The transcript below is generated from the committed JSON, not edited terminal
output:

```console
$ python benchmarks/summarize.py
ModelPact R1 executed evidence
ForkBench status=PASS witnesses=5 clusters=5 rank=2 factor_bytes=1292 target=1.000 unselected_rejection=1.000 holdout=PASS
R1Loop status=PASS naive=SEMANTIC_CONFLICT merge=SEMANTIC_MERGE_VERIFIED rebase=SEMANTIC_REBASE_VERIFIED revert=VERIFIED_LOGICAL_STACK_RECONSTRUCTED
ClosureMatrix status=PASS model=TinyCausalLM subsets=63/63 failures=8 coverage=EXHAUSTIVE_SUBSETS
BenignCollusion status=PASS model=TinyCausalLM subsets=40/63 active_found=true pairwise_found=false minimal_order=3
HuggingFaceLocal status=PASS patches=2 compose=COMPOSITION_CLOSED rebase=DIRECT_TRANSPLANT_VERIFIED standalone=2/2
Materialization strategy=planned-output-shard shards=6 read_bytes=79129 write_bytes=48559 peak_rss=None
CEGIS status=PASS negative_result=true search_failures=3->4
Environment python=3.14.0 torch=2.13.0+cpu gpu_executed=false
```

The raw records are in [`research/artifacts`](research/artifacts). The committed
[`forkbench-run`](research/artifacts/forkbench-run) contains the matching base
checkpoint, scoped diff bundle, complete patch, certificate, prompt hashes,
compile/search/validation/holdout evidence, and standalone apply/verify tools.

## What ModelPact investigates

Model updates are usually distributed as checkpoints, deltas, or adapters. Those
formats say how parameters changed, but not precisely which behavior the change
claims, what it must preserve, whether composition remains valid, or whether a
new base still supports it.

ModelPact treats a learned change as a scoped software artifact:

```python
BehaviorPatch(
    base_signature=...,
    delta_program=...,
    provides=...,
    preserves=...,
    requires=...,
    lineage=...,
    evidence=...,
)
```

The research question is whether that abstraction makes behavioral diff,
constrained compilation, composition testing, semantic repair, audit, rebase,
and reversion more reproducible. ModelPact proves nothing beyond the contracts,
probe spaces, policies, environments, and budgets that were actually executed.

## Install

ModelPact requires Python 3.11 or newer and PyTorch. Normal tests and the tiny
workflow do not require a network connection.

```console
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,research]"
```

On Windows, use `.venv\Scripts\python`. Add the local Hugging Face integration
without downloading a model:

```console
python -m pip install -e ".[dev,research,huggingface]"
```

Run the retained tiny experiment:

```console
modelpact benchmark forkbench \
  --artifacts artifacts/forkbench \
  --output artifacts/forkbench-result.json \
  --json
```

## Behavior Contract v1

Optimization objectives and acceptance assertions are intentionally separate.
A discrete JSON or generation assertion is not presented as differentiable.

```yaml
schema_version: 1
id: json-mode
model_requirements:
  output_semantics: causal_lm
compile:
  objectives:
    - id: imitate-json
      type: teacher_kl
      source: probes/train.jsonl
verify:
  targets:
    - id: parses
      type: json_parse
      source: probes/validation.jsonl
      minimum_pass_rate: 0.98
  guards:
    - id: preserve-base
      type: base_kl
      source: guards/validation.jsonl
      maximum_mean: 0.02
holdout:
  sealed: true
  targets: holdout/targets.jsonl
  guards: holdout/guards.jsonl
  unseal_policy: final_candidate_only
generation:
  mode: greedy
  max_new_tokens: 128
```

Implemented compile objectives include teacher cross-entropy/KL, preferred
sequence margin, base KL, hidden-state matching, and activation directions.
Assertions cover token/sequence probability, margins, multiple choice, exact and
normalized matches, bounded regular expressions, JSON parsing/schema, free
generation, reference/base KL, generation length, and perplexity. See
[`docs/contracts.md`](docs/contracts.md).

## Compilation and selective extraction

The compiler scores candidate linear modules using target and guard gradients,
builds a contrastive low-rank basis, and optimizes factors with explicit
primal-dual guard multipliers. It tracks the best exactly re-evaluated feasible
candidate instead of silently replacing constraints with one weighted loss.

Generic `compile` supports bounded deterministic CEGIS over visible probe
mutations. Extraction uses a target model only on the selected witness domain and
the base model as preservation teacher elsewhere. Compile, search, validation,
and sealed holdout roles are globally disjoint. Module and rank removal claims
are emitted only for candidates actually executed within the declared budget.

The optimizer exposes both contrastive-gradient and target-checkpoint-delta
initialization. Automatic synthesis in R1 is low-rank for linear weights; the
runtime IR is broader than the compiler (see Limitations).

## Free-generation and independent verification

Final generative assertions run autoregressive decoding. Evidence records the
policy, seed, prompt/output/token hashes, parser result, and item-level metrics.
Holdout capability is granted only after a final candidate ID is selected.

`modelpact verify` re-hashes the bundle, independently mounts the delta, executes
every embedded target and preservation contract, and creates a new certificate.
An external `--policy` adds checks; it cannot replace bundled claims.

Generated `verify_patch.py` does not import ModelPact. Reviewed built-in Tiny and
local Hugging Face verification paths are embedded in the generated script;
custom adapters must be separately trusted code outside the untrusted bundle.

## Composition closure and semantic merge

R1 deltas are additive:

```text
Delta(p + q) = Delta(p) + Delta(q)
```

The parameter operation is commutative and associative. Behavioral validity is
not. `compose` evaluates the base, each singleton, and the union, then classifies
closure from executed contract margins. It records module/sparse overlap,
low-rank principal angles, baseline margins, and semantic interaction residuals.

When addition fails, `merge` builds the union objective, initializes from parent
deltas, and runs a fresh tiny-model optimization. A successful merge is a new,
complete Behavior Patch Bundle with both parents in its lineage—not parameter
averaging relabeled as semantic repair.

## Higher-order audit

`audit` always executes proposed subsets. Its sparse degree-1/2/3 pseudo-Boolean
surrogate is a search heuristic, never a certificate. It supports exhaustive
small pools, budgeted active selection, dependency closure, and executed ddmin
reduction of failing subsets.

The committed six-patch TinyCausalLM benchmark exhaustively executed all 63
subsets and found eight failures, exactly the stacks containing the same three
colluding patches. Every singleton and pair passed. The active auditor executed
40 subsets and used ddmin to recover the exact triple; executed contract outcomes,
not its surrogate, determined failure.

## Semantic rebase and revert

Rebase first attempts physical transfer only when schemas permit it, then
executes the original and new-base contracts. A passing transplant is labeled
`DIRECT_TRANSPLANT_VERIFIED`. A failing Tiny transfer triggers behavioral
recompilation using the old patched model as target teacher and the new base as
preservation teacher.

Stack resolution pins base, patch, contract, audit, certificate, policy, and
resolved-artifact hashes. Removing a patch reconstructs and verifies the logical
remainder. A nonempty additive reconstruction is reported as
`VERIFIED_LOGICAL_STACK_RECONSTRUCTED`; it is not called an exact runtime unmount,
numeric inverse, or semantic recompile.

## CLI

The installed `modelpact` command exposes:

```text
scan       diff       contract   compile    extract
inspect    apply      verify     compose    merge
audit      rebase     revert     resolve    emit
benchmark
```

Commands use deterministic JSON ordering, bounded resource arguments, nonzero
failure exits, and distinct `FAIL`, `INCONCLUSIVE`, and `UNSUPPORTED` outcomes.
Trusted custom adapters use `module:attribute`; only load code you trust.

Useful entry points:

```console
modelpact contract validate contract.yaml --json
modelpact scan --model tiny --checkpoint models/base --output base.json --json
modelpact compile --base tiny --checkpoint models/base --spec contract.yaml \
  --output patches/example --cegis-rounds 2 --json
modelpact verify patches/example --base models/base --adapter tiny --json
modelpact compose --base models/base patches/a patches/b --output composed --json
modelpact audit --base models/base --patch-dir patches --subset-budget 100 --json
modelpact emit verify patches/example --output verify_patch.py
```

`apply --mode materialize` writes a new checkpoint. `apply --mode runtime`
executes a real parametrized mount/unmount validation and writes a deterministic
runtime-stack descriptor; the in-memory session is explicitly ephemeral because
a CLI process cannot leave its Python model object alive after exit. Library
users mount persistent in-process sessions with `modelpact.patch.mount`.

## Patch Bundle v1

```text
patch/
├── manifest.json
├── delta-program.json
├── tensors.safetensors
├── contracts/
├── probes/
├── evidence/
├── certificate.json
├── report.md
├── apply_patch.py
└── verify_patch.py
```

Delta Program v1 is a safe typed additive AST: low-rank matrices, sparse
matrices, vectors, aliases, and sums. There is no pickle, `eval`, lambda, or
serialized callable. Tied parameter aliases are explicit and validated.

Identity has three layers: `patch_id` identifies the executable delta/contracts/
base/lineage; `evidence_id` additionally binds probes, evidence, and reports;
`bundle_id` addresses the final manifest including certificate and generated
helpers. This avoids certificate hash cycles without leaving evidence mutable
under the same evidence identity. The normative format is in [`SPEC.md`](SPEC.md).

## Supported configurations

- Internal deterministic decoder-only `TinyCausalLM` and tokenizer.
- Local Hugging Face decoder-only causal LMs with SafeTensors, bounded local
  metadata, no remote code, and no automatic downloads.
- Trusted local PyTorch adapters implementing the narrow typed protocol.
- Runtime/fold support for linear weights and biases, embeddings/output heads,
  one-dimensional scale vectors, sparse residuals, tied state, and aliases.
- CPU execution and a single CUDA device. The committed release evidence is CPU
  only; no GPU result is claimed.

## ModelPactBench results

All values below come from committed machine-readable artifacts. One deterministic
seed is integration evidence, not a statistical superiority claim.

| Experiment | Executed result | Evidence boundary |
| --- | --- | --- |
| ForkBench | 5 witnesses/clusters; selected rank 2; target 1.000; unselected rejection 1.000; sealed holdout PASS | Real Tiny LM, finite probes, one seed |
| Unified R1 loop | Additive conflict; fresh merge PASS; direct rebase FAIL; semantic rebase PASS; logical revert PASS | Real Tiny LM, finite probes, no sealed holdout |
| Closure Matrix | 63/63 subsets; 8 failing; all singletons/pairs PASS | Six real TinyCausalLM BehaviorPatch bundles |
| Benign Collusion | active search found and minimized an order-3 failure in 40/63 subsets; pair-only missed | Real runtime-mounted TinyCausalLM stacks |
| Hugging Face local | 2 patches, closed composition, direct verified rebase, 2/2 package-independent verifiers PASS | Generated local 1-layer GPT-NeoX, one seed |
| Locality/CEGIS | search failures increased 3 to 4 | Negative polynomial-organism result |

On the recorded 12-logical-CPU Windows host, ForkBench took 18.07 s (14.12 s
extraction); the Hugging Face workflow took 17.39 s (3.69 s compiling 360 total
steps). The committed streaming materialization read 79,129 bytes, wrote 48,559
bytes, and took 0.036 s before its manifest; peak RSS was unavailable on this
Windows host. These timings are local observations, not cross-platform
performance claims. See [`research/RESULTS.md`](research/RESULTS.md) and the raw
JSON.

Notable negative results:

- The analytic CEGIS organism became worse, not better (3 search failures to 4).
- Weighted addition and TIES also passed the finite scalar merge contract; that
  experiment does not establish ModelPact superiority.
- ForkBench's worst validation-item base KL was about 1.956 despite passing its
  declared 4.0 item/2.5 mean thresholds.
- The active collusion search used more executions than deterministic random and
  parameter-overlap baselines in this one run; only the active path performed
  the reported ddmin reduction.
- No GPU run or multi-seed significance analysis has been performed.

## Architecture

The core layers are intentionally independent of CLI rendering:

```text
trusted adapter + model manifest
            │
contract AST ──> objectives/assertions/holdout gate
            │
diff/search ──> compiler + CEGIS + minimizer
            │
typed delta program ──> runtime mount / checkpoint fold
            │
verify ──> certificate
            │
compose / merge / audit / rebase / stack resolution
```

[`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md) gives the algorithms and equations;
[`research/PROTOCOL.md`](research/PROTOCOL.md) preregisters H1–H6 and their
falsification conditions.

## Trust model

Adapters are arbitrary local Python and are trusted. Contracts, manifests,
delta programs, tensors, certificates, checkpoint indexes/metadata, probe
records, output paths, and lockfiles are untrusted data.

ModelPact rejects traversal and symlinks across bundle/checkpoint boundaries,
duplicate JSON keys, nonfinite values, excessive nesting/counts/sizes, unknown
delta operations, malformed aliases, pickle weights, remote model code, and
bundle-local adapter imports. Writes use same-filesystem temporary paths and do
not overwrite source checkpoints. Details are in
[`docs/trust-model.md`](docs/trust-model.md).

## Exact limitations

- Verification is finite, sampled, or search-budgeted. It is not formal
  verification of open-ended language behavior.
- Automatic compilation currently synthesizes low-rank linear-weight deltas.
  Sparse/vector/bias/embedding operations are implemented for IR application,
  but not selected automatically by the compiler.
- Generic semantic repair/recompile paths are implemented for Tiny models;
  arbitrary custom/Hugging Face conflicts can return `UNSUPPORTED` rather than
  silently falling back to averaging.
- The retained exhaustive closure/collusion ground truth uses six analytically
  compiled low-rank TinyCausalLM patches. It is still a controlled synthetic
  model organism, not evidence for large-model audit discovery rates.
- Some gradient, activation, and raw-output interaction diagnostics are emitted
  as `NOT_AVAILABLE` when their required evidence was not collected. Executed
  contract margins remain authoritative.
- Cross-architecture rebase requires compatible tokenizer/output semantics and
  recompiles behavior; it does not map tensors between architectures.
- Materialization is metadata-planned and streams one output shard at a time,
  but patch factors, one target delta, a SafeTensors snapshot copy, and one
  output shard remain resident. A tensor larger than the shard target remains
  indivisible; no constant-memory or cross-platform peak-RSS bound is claimed.
- R1 excludes quantized training, MoE, tensor-parallel checkpoints, distributed
  compilation, multimodal/diffusion models, remote API models, RLHF/DPO/GRPO,
  signing infrastructure, and a hosted registry.
- Windows was used for the committed local evidence. Linux is the CI target; a
  passing remote workflow is reported only after it actually runs.

## Related work and novelty boundary

ModelPact does not claim priority for model editing, patching, differential
testing, task vectors, model merging, LoRA composition/transfer, rollback,
checkpoint version control, mechanistic diffing, or contract testing. Its
defensible contribution is the integrated executable loop and evidence-bearing
artifact. The 2026-08-15 literature/repository review is in
[`research/related-work.md`](research/related-work.md) and must be rerun on the
actual release date.

The ModelPactBench name was searched on 2026-08-15 for obvious research/software
collisions; that is not a trademark or legal-availability claim.

## Reproduce and contribute

```console
python -m ruff format --check src tests benchmarks
python -m ruff check src tests benchmarks
python -m mypy src/modelpact
python -m pytest
python -m build
```

Larger/manual workflows live in [`.github/workflows/research.yml`](.github/workflows/research.yml).
Please preserve unsuccessful runs and report baselines that outperform ModelPact.
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Roadmap

R1 deliberately stops before preference/RL patches, activation/prefix backends,
quantized application, signing, distributed/tensor-parallel/MoE compilation,
multimodal or cross-tokenizer rebase, and a public registry. Those belong to
later research phases only after the single-device contract loop is better
validated.
