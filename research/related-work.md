# Related work and novelty boundary

Literature and repository check: 2026-08-15. This search must be repeated on the
actual release date.

ModelPact does not claim priority for model editing, patching, differential
testing, task vectors, model merging, LoRA composition or transfer, rollback,
checkpoint versioning, mechanistic diffing, or contract testing. Its research
question concerns the integration and execution of behavioral diff, a contract
language, constrained patch compilation, contract-closed composition,
higher-order audit, semantic merge/rebase, and evidence-bearing bundles.

## Editing and steering

- [EasyEdit](https://arxiv.org/abs/2308.07269) provides a framework spanning
  multiple knowledge-editing methods. [EasyEdit2](https://arxiv.org/abs/2504.15133)
  broadens this toward test-time steering vectors. ModelPact is neither a wrapper
  nor a replacement; its unit is a contract-carrying artifact and its focus is
  cross-operation evidence.
- [The Mirage of Model Editing](https://arxiv.org/abs/2502.11177) shows why
  teacher-forced edit metrics can overstate deployment performance. This directly
  motivates ModelPact's mandatory autoregressive verification and negative-result
  handling.
- [Patching LLM Like Software](https://arxiv.org/abs/2511.08484) learns compact
  prefix policy patches. It establishes important patching prior art; R1 instead
  studies additive parameter programs with preservation and composition
  contracts.
- [Hybrid-Policy Self-Editing](https://arxiv.org/abs/2608.11660) targets
  composable use of unstructured edited knowledge through hybrid-policy
  self-distillation. Its “composable” target is reasoning over injected facts,
  distinct from ModelPact's closure of independently distributed patch contracts.
- Lifelong editing systems include
  [ELDER](https://ojs.aaai.org/index.php/AAAI/article/view/34622),
  [MEMOIR](https://openreview.net/forum?id=t94tALZvZE), and
  [SimIE](https://mlanthology.org/icml/2025/guo2025icml-lifelong/). They motivate
  sequential-retention baselines; ModelPact does not claim to solve lifelong
  knowledge editing generally.

## Behavioral and representational diff

- [BehaviorBox](https://arxiv.org/abs/2506.02204) automatically discovers
  fine-grained contexts where model performance differs.
- [Behavioral Shift Auditing](https://arxiv.org/abs/2410.19406) detects output
  distribution shifts with sequential hypothesis testing and false-positive
  control.
- [Delta-Crosscoder](https://arxiv.org/abs/2603.04426) isolates changed latent
  directions in narrow fine-tunes;
  [cross-architecture crosscoders](https://arxiv.org/abs/2602.11729) extend
  unsupervised model diffing across architecture families.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
  provides activation access and interventions. ModelPact's projected activation
  fingerprints can use analogous evidence but do not purport to reverse engineer
  a complete learned algorithm.

ModelPact witnesses and clusters are explicitly scoped empirical objects. They
do not compete with or replace mechanistic explanations.

## Parameter-efficient transfer and composition

- [PEFT](https://github.com/huggingface/peft) is the major practical adapter
  library and supports LoRA training/application. A Behavior Patch can contain
  low-rank factors, but adds contracts, identity, lineage, independent evidence,
  composition closure, and rebase semantics.
- [Trans-LoRA](https://arxiv.org/abs/2405.17258) and
  [LoRASuite](https://arxiv.org/abs/2505.13515) transfer adapters across base
  changes. They are required rebase comparisons where reproduction is feasible.
- [SCALE-LoRA](https://arxiv.org/abs/2605.01429) audits post-retrieval LoRA
  composition and studies residual merging plus multi-view reliability.
- [Colluding LoRA](https://arxiv.org/abs/2603.12681) demonstrates that adapters
  that appear benign alone can fail when composed. ModelPact's harmless public
  collusion benchmark and higher-order audit address this class of combinatorial
  blindness without using harmful request datasets.

## Model merging and checkpoint history

- [Git-Theta](https://arxiv.org/abs/2306.04529) tracks structured parameter
  updates through a Git extension. ModelPact uses Git for source/artifact history
  and does not recreate its object database or transport semantics.
- [mergekit](https://github.com/arcee-ai/mergekit) and its
  [system paper](https://arxiv.org/abs/2403.13257) implement a broad, practical
  collection of model-merging techniques. ModelPact keeps such algorithms as
  baselines; a semantic merge is defined by new optimization and union-contract
  execution.
- [TIES-Merging](https://arxiv.org/abs/2306.01708) resolves magnitude and sign
  interference. [CAT Merging](https://arxiv.org/abs/2505.06977) projects or masks
  conflict-prone task-vector components.
- [Task-level merging collapse](https://arxiv.org/abs/2603.09463) reports that
  representation incompatibility can be more predictive than parameter conflict,
  reinforcing the rule that overlap is diagnostic rather than authoritative.
- [MergeBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/91f7f71cb04699f387dc863da42a1fe3-Abstract-Datasets_and_Benchmarks_Track.html)
  evaluates domain-specialized LLM merging. ModelPactBench instead targets selective
  extraction, exact small-pool closure ground truth, higher-order failure search,
  semantic repair, and rebase.

## Defensible contribution statement

Subject to the preregistered experiments, ModelPact contributes an integrated
abstraction and executable loop:

```text
behavioral diff + contract language + constrained patch compilation
+ contract-closed composition + higher-order audit
+ semantic merge/rebase + independently regenerable evidence
```

The experiments test whether that integration is useful. They do not assume it
is universally feasible, safe, minimal, architecture-independent, or superior to
specialized methods.
