# ModelPactBench R1 results

Execution date: 2026-08-15. These are development-release results from the
committed machine-readable artifacts in [`research/artifacts`](artifacts), not
values copied from the design handover. No GPU experiment was run.

## Environment

The retained local run used CPython 3.14.0, PyTorch 2.13.0+cpu, and 12 logical
CPUs on Windows 11. CUDA was unavailable. Exact package versions are in
[`environment.json`](artifacts/environment.json). The normal CI targets Linux
and Python 3.11, but that outcome must be reported from the remote checks,
not inferred from this local run.

## Result summary

| Experiment | Terminal result | Executed evidence | Scope |
| --- | --- | --- | --- |
| ForkBench | PASS | 5 witnesses and 5 clusters; rank-2 selected patch; target 1.000; unselected rejection 1.000; unchanged controls 1.000; holdout PASS | Tiny causal LM, one seed, finite/search-audited probes |
| Unified R1 loop | PASS | two singletons PASS; additive stack conflict; fresh semantic merge PASS; direct rebase FAIL; semantic rebase PASS; logical revert PASS | Tiny causal LM, one seed, finite probes |
| Closure Matrix | PASS | all 63 subsets executed; 8 failing subsets; all singletons/pairs pass | Six real low-rank TinyCausalLM BehaviorPatch bundles |
| Benign Collusion | PASS | order-3 failure found and minimized; pair-only missed; 40/63 subsets executed | Runtime-mounted TinyCausalLM stacks, active budgeted |
| Semantic Merge | PASS | parent singletons PASS; naive sum FAIL; joint compile PASS | Scalar PyTorch organism |
| RebaseBench | PASS | same-family direct FAIL/recompile PASS; cross-architecture behavioral compile PASS | Two-feature PyTorch organism |
| Locality/CEGIS | PASS as experiment, negative treatment result | failures changed 3 to 4 | Polynomial PyTorch organism |
| Hugging Face local | PASS | two patches and holdouts PASS; composition closed; direct rebase verified; 2/2 package-independent verifiers PASS | Generated local one-layer GPT-NeoX, one seed |

“PASS as experiment” means the benchmark executed its preregistered checks and
faithfully recorded its outcome. It does not mean the tested method improved the
metric.

## ForkBench: selective extraction

The base and multi-change target were real trained `TinyCausalLM` checkpoints.
The target changed five synthetic domains: a fact, output format, response style,
symbolic rule, and structured choice. Differential execution found five scoped
witness clusters; four of five witnesses were delta-debug minimized.

The selected fact cluster produced a patch with:

- two active modules (`layers.0.mlp.down_proj` and tied `lm_head`), rank one each;
- 1,292 factor-tensor payload bytes and a 597,601-byte complete evidence bundle;
- three compile attempts and three executed minimization candidates;
- `MODULE_ONE_MINIMAL` and `RANK_LOCAL_MINIMUM` within that executed budget;
- 24 free-generation records;
- 1.000 selected transfer, 1.000 unselected-change rejection, and 1.000 unchanged
  control preservation;
- sealed target and guard holdout PASS, opened only after the patch ID existed;
- exact runtime unmount restoration.

The package-independent `verify_patch.py` was run with `python -S -P`, with
`modelpact` absent from its import path. It independently passed target, guard,
free-generation, reference/base distribution, and holdout assertions. Its full
new report is [`forkbench_standalone_verify.json`](artifacts/forkbench_standalone_verify.json).

Negative finding: the worst validation-item base KL was 1.9561. It passed the
declared maximum-item 4.0 and mean 2.5 thresholds, but this is substantial drift
and should not be hidden by the 1.000 generation pass rates.

Raw result: [`forkbench.json`](artifacts/forkbench.json).
The complete reproducible run—including the internal base checkpoint, diff
bundle, Behavior Patch Bundle v1, and standalone tools—is retained under
[`forkbench-run`](artifacts/forkbench-run).

## Real Tiny conflict, merge, rebase, and revert

The unified loop compiled a fact patch and a format patch independently. Both
passed their finite contracts. Their true additive union emitted `r` for the
format prompt instead of `{`, yielding `SEMANTIC_CONFLICT`.

A fresh 100-step union-contract optimization produced a delta different from the
parent sum and passed both targets plus guards (`SEMANTIC_MERGE_VERIFIED`). For
rebase, base-v2 learned `F:b -> D`. Direct fact-patch transplant was executed and
failed. A 120-step behavioral recompile restored `F:a -> G` while retaining
`F:b -> D`, its variants, and the separate `P -> X` new-base control.

Removing the format patch reconstructed and verified the remaining fact stack.
The result is labeled `VERIFIED_LOGICAL_STACK_RECONSTRUCTED`; no floating-point
inverse or exact runtime unmount is claimed.

Raw result: [`r1_loop.json`](artifacts/r1_loop.json).

## Contract Closure Matrix

Six deterministic, content-addressed low-rank patches were compiled against one
TinyCausalLM base. The audit executed the empty stack plus every one of the 63
nonempty subsets through runtime mounting, logits, greedy generation, and exact
unmount checks. Every singleton and pair passed; eight subsets failed, exactly
those containing `behavior-0 + behavior-1 + behavior-2`. Only this run emits
`EXHAUSTIVE_COMPOSITION_AUDIT` / `EXHAUSTIVE_SUBSETS`.

The six patches and ordered subset margins/output hashes are deterministic
across repeated executions; elapsed time is intentionally measured rather than
canonicalized. This controlled tiny model establishes exact causal-LM ground
truth, but cannot establish discovery performance for large language models.

Raw result: [`closure_matrix.json`](artifacts/closure_matrix.json).

## Benign Collusion

Every singleton and pair passed. Active search found a failing six-patch stack
at execution 22 of its 40-of-63 budget, and executed ddmin reduced it to the
one-minimal triple `behavior-0 + behavior-1 + behavior-2`. Singleton-only and
pairwise-only baselines found nothing and therefore had false-assurance rate 1.0.

The deterministic random baseline found a failing four-patch superset at
execution 2 and parameter-overlap ordering found a six-patch failing superset at
execution 1; each was faster than active search in this seed. The active path's
distinct result is the executed ddmin recovery, not superior time to first
failure. H4's efficiency claim is therefore not supported.

Raw result: [`collusion.json`](artifacts/collusion.json).

## Semantic merge baselines

The scalar parent patches passed independently and naive addition failed. A
120-step ModelPact joint optimization passed. So did joint multitask low-rank
training. Weighted addition and TIES also passed this finite contract with no
optimization; DARE, task arithmetic, CAT-style projection, and naive sum failed.

This is explicitly a negative result for any blanket superiority claim. It shows
that ModelPact can execute semantic recompilation, not that recompilation is
always necessary or best.

Raw result: [`merge.json`](artifacts/merge.json).

## RebaseBench

The same-family two-feature organism executed direct transfer, observed failure,
and passed after 100 optimization steps while retaining its old and new-base
guards. The controlled cross-architecture organism skipped physical transplant
and passed behavioral recompilation.

The real unified Tiny loop provides stronger causal-LM evidence for failed-direct
then semantic rebase. The local Hugging Face case instead found direct transfer
already valid; it correctly emitted `DIRECT_TRANSPLANT_VERIFIED` and did not
pretend a recompile occurred.

Raw results: [`rebase.json`](artifacts/rebase.json),
[`r1_loop.json`](artifacts/r1_loop.json), and
[`huggingface_local.json`](artifacts/huggingface_local.json).

## Locality and CEGIS

The polynomial locality organism began with three search failures and ended with
four after four CEGIS rounds. `search_failures_reduced` is false and
`negative_result` is true. H6 is therefore not supported by this organism.

ForkBench separately shows that CEGIS can find and insert real target/guard
counterexamples before a final candidate passes holdout, but this release lacks
a matched multi-seed fixed-probe control for that exact Tiny task. That evidence
is descriptive, not a causal comparison.

Raw result: [`cegis.json`](artifacts/cegis.json).

## Local Hugging Face integration

The fully offline benchmark generated and saved a one-layer GPT-NeoX checkpoint
and real tokenizer locally, then reloaded through the built-in safe adapter.
There were no third-party weights or network calls.

- Patch A changed `fact_a` to `X`; patch B changed `fact_b` to `Y`.
- Both validation and sealed holdout suites passed and runtime unmount was exact.
- Additive composition generated `fact_a=X`, `fact_b=Y`, `control=C` and was
  classified `COMPOSITION_CLOSED`.
- Direct transplant of A onto base-v2 was behaviorally verified while preserving
  base-v2's `control=Y` change.
- Both generated standalone verifiers ran under `python -S -P`, with ModelPact
  unimportable and the reviewed embedded Hugging Face adapter selected. Both
  passed target, guard, and holdout roles.

This fixture is tiny and one-seed. It establishes local integration mechanics,
not compatibility with arbitrary Transformers architectures.

Raw result: [`huggingface_local.json`](artifacts/huggingface_local.json).

## Performance observations

| Stage | Wall time | Other executed quantity |
| --- | ---: | --- |
| ForkBench total | 33.64 s | 200 base + 150 target training steps |
| ForkBench training | 4.49 s | six setup behaviors checked |
| ForkBench diff | 0.11 s | 6 prompts, 49 tokens, 5 witnesses |
| ForkBench extraction | 26.70 s | 3 attempts, bounded CEGIS |
| ForkBench minimization | 0.24 s | 3 candidates |
| Closure Matrix | 1.17 s | empty stack + all 63 nonempty TinyCausalLM subsets |
| Benign Collusion | 19.83 s | 40 active subsets plus exact ground truth and baselines |
| Hugging Face total | 17.39 s | four locally trained checkpoints |
| Hugging Face compilation | 3.69 s | 360 total compiler steps |
| Streaming materialization | 0.046 s before manifest | 79,129 bytes read; 48,559 bytes written; 6 output shards |

These are single observations on the environment above. No GPU memory, peak-RSS
comparison, throughput scaling, or statistical timing interval is claimed.
The Windows runtime reported peak RSS as explicitly unavailable. With a 5,000-
byte shard target, the largest indivisible tensor/shard payload was 16,576 bytes;
the writer records that exception instead of claiming the requested bound was
strictly met. The generated checkpoint and manifest are retained under
[`materialization-run`](artifacts/materialization-run).

## Hypothesis disposition

| Hypothesis | R1 disposition |
| --- | --- |
| H1 selective extraction | Supported on one Tiny seed within finite controls; multi-seed comparison pending |
| H2 contract-aware merge | Functional success shown; superiority not established because several baselines passed |
| H3 interaction diagnostics | Inconclusive; no adequately powered predictive comparison |
| H4 higher-order audit | Failure found, but efficiency claim not supported; random was earlier in this run |
| H5 semantic rebase | Supported descriptively by the Tiny failed-direct/recompile case; multi-seed baseline comparison pending |
| H6 CEGIS | Negative on polynomial organism; controlled Tiny comparison pending |

## Release evidence boundaries

- One local seed was run for trained-model benchmarks; no significance claim is
  permitted.
- Closure/collusion exact ground truth uses a controlled one-layer TinyCausalLM,
  not a large-model patch pool.
- No GPU run occurred.
- Some requested baselines remain algorithmic organisms rather than full
  third-party reproductions.
- ModelPact may return `UNSUPPORTED` or `INCONCLUSIVE` instead of weakening a
  contract or extrapolating beyond executed evidence.
- Failed and unfavorable results above are intentionally retained.
