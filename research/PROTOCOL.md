# ModelPactBench preregistered research protocol

Protocol version 1, recorded before release benchmark results are interpreted.
All seeds, raw prompt-level results, environment manifests, and unsuccessful runs
are retained. A single seed cannot support a superiority claim. Paired bootstrap
intervals use deterministic seeds; multiplicity-adjusted tests are reported only
when significance language is used.

The benchmark suite was renamed from PactBench to ModelPactBench after a
namespace check on 2026-08-15 found existing public uses of PactBench/PACTBench.
This is a practical collision-avoidance decision, not a trademark opinion or
legal clearance. The namespace must be checked again at the actual release date.

This protocol describes the experiments required to evaluate H1–H6, not claims
already established by the small deterministic CI organisms. Multi-seed success
criteria and baseline comparisons remain unmet until the corresponding raw runs
are committed and analyzed. Analytic merge/audit/rebase organisms establish
control-flow behavior only and cannot by themselves support language-model or
method-superiority claims.

## H1 — Selective extraction

- Hypothesis: contract-guided extraction transfers a selected ForkBench behavior
  cluster while importing less nonselected target behavior than rank-matched SVD
  of the full target delta.
- Primary metrics: selected holdout pass rate and mean nonselected base KL.
- Baselines: full target delta, rank-matched target-delta SVD, standard LoRA,
  gradient-saliency LoRA, random-module LoRA.
- Success criterion: across at least three training seeds, ModelPact is no worse
  on selected holdout and has lower paired nonselected drift, with the complete
  confidence interval reported.
- Falsification: lower selected retention, indistinguishable/worse import drift,
  or no compact passing patch within the predeclared rank/module budget.
- Budget: tiny CPU runs use at most 12 modules, rank 16, five CEGIS rounds, and
  the committed step budget; larger local/GPU runs are separate strata.
- Statistics: prompt-paired bootstrap, 2,000 samples, seed 81273; Holm correction
  across the two primary endpoint comparisons if p-values are reported.

## H2 — Contract-aware merge

- Hypothesis: semantic merge retains union contracts more reliably than naïve
  addition under equal or explicitly normalized compute budgets.
- Benchmark: Semantic Merge plus conflicting Closure Matrix subsets.
- Metrics: union target/guard closure, worst margin, patch bytes, forward/backward
  tokens and steps.
- Baselines: naïve/weighted sum, task arithmetic, TIES, DARE, CAT-style local
  baseline, and joint multitask LoRA.
- Success criterion: higher multi-seed closure rate without weakening any parent
  threshold; compute and complexity are reported alongside.
- Falsification: a baseline matches/exceeds closure under its documented budget,
  semantic merge breaks a parent contract, or conflicts remain empirically
  infeasible.
- Budget/statistics: equal optimization-step and rank budgets where meaningful;
  Wilson/paired bootstrap intervals for closure and margins.

## H3 — Interaction diagnostics

- Hypothesis: behavioral and representation diagnostics predict composition
  failure better than raw parameter overlap alone.
- Benchmark: exact 63-subset Closure Matrix across multiple model seeds.
- Metric: held-out subset AUROC/AUPRC and calibration error; parameter-overlap
  ranking is the primary baseline.
- Success criterion: bootstrap interval for AUPRC improvement excludes zero.
- Falsification: parameter overlap ties or wins, or improvements disappear under
  held-out patch-family/model seeds.
- Budget: all 63 subsets per six-patch pool; no surrogate labels.
- Statistics: pool/seed-blocked bootstrap and Holm correction across diagnostic
  families.

## H4 — Higher-order audit

- Hypothesis: active sparse-interaction search discovers higher-order failing
  subsets with fewer executions than random subset testing.
- Benchmark: Benign Collusion pools with exact exhaustive ground truth.
- Metrics: failure discovery recall, executions/time to first failure, minimal
  failing-subset recovery, and false assurance rate.
- Baselines: singleton only, pairwise only, random, parameter/subspace/gradient
  rankings.
- Success criterion: at matched budget, active search has higher mean recall and
  lower median executions to first failure across seeds.
- Falsification: random matches/wins, the active search misses known failures, or
  uncertainty acquisition consumes the budget without useful coverage.
- Budget: committed subset budgets below exhaustive size; exhaustive evaluation
  is run separately only to establish ground truth.
- Statistics: paired seed/pool bootstrap; every missed failure remains visible.

## H5 — Semantic rebase

- Hypothesis: contract-guided rebase preserves both source-patch behavior and
  new-base improvements better than direct transplant.
- Benchmark: RebaseBench same-family and controlled cross-architecture cases.
- Metrics: original target retention, new-base control retention, guard drift,
  patch bytes, steps, and tokens.
- Baselines: direct copy, equal-budget retraining, teacher distillation without
  guards, and optional separately reproduced Trans-LoRA/LoRASuite.
- Success criterion: improved joint retention frontier over direct copy and
  unguarded distillation across seeds.
- Falsification: direct transfer wins, guard-constrained recompilation erases
  new-base gains, tokenizer/output semantics block the transfer, or patch growth
  exceeds the declared cap.
- Statistics: paired bootstrap by prompt within seed and seed-level effect
  distribution; cross-architecture results are a separate stratum.

## H6 — Counterexample-guided compilation

- Hypothesis: CEGIS reduces unseen target and guard failures relative to a
  fixed-probe compiler.
- Benchmark: Locality and CEGIS with paraphrases, neighbors, distractors,
  formatting perturbations, and untouched sealed holdout.
- Metrics: fuzzer failure rate before/after, holdout target/guard pass rates,
  worst drift, patch complexity, and search/optimization cost.
- Baseline: identical compiler configuration with zero CEGIS rounds.
- Success criterion: lower paired failure rate without worse sealed guard margin.
- Falsification: no improvement, regression on holdout/guards, repeated holdout
  tuning, or materially larger patches outside the declared budget.
- Statistics: prompt-paired bootstrap with separate reporting of every seed where
  CEGIS does not help.

## Reporting and stopping

Validation is used for candidate selection. A sealed holdout is opened once for
the final candidate. Failure ends that contract version. Current CPU runners
stop at explicit optimization-step, CEGIS-round/search, or subset budgets and
record wall time; they do not implement a general in-process wall-time or memory
interrupt. Larger experiments must declare any external scheduler limits and
record peak memory separately. `INCONCLUSIVE` remains distinct from pass.
Benchmark names and availability are rechecked before publication.
