# ModelPact: Contract-Carrying Learned Behavior Patches

## Abstract

ModelPact studies whether a learned behavioral change can be distributed as a
scoped software artifact rather than as an opaque checkpoint or parameter delta.
The artifact combines a safe additive delta program with executable target and
preservation contracts, lineage, and independently regenerable evidence. The
delta IR can represent low-rank matrices, sparse matrices, and vectors; the R1
reference compiler currently synthesizes low-rank deltas for linear weights.
The system does not infer complete model semantics and does not certify
open-ended safety. It executes finite, sampled, and searched tests whose scope
is recorded explicitly.

## Behavioral artifact model

Let a decoder-only model be (f_\theta). An R1 patch (p) carries an additive
delta (\Delta_p):

\[
f_{\theta \oplus p}=f_{\theta+\Delta_p}.
\]

The parameter expression alone says nothing about intent. A Behavior Patch also
carries base identity, target and preservation contracts, probe/holdout
identities, dependencies and lineage, and the evidence produced by executing
those contracts. The adapter that instantiates a model is trusted local code;
the patch and every associated document remain untrusted data.

A contract is

\[
C=(O,A,G,H,\Pi),
\]

where (O) contains differentiable compilation objectives, (A) target
acceptance assertions, (G) preservation guards, (H) a sealed-holdout policy,
and (\Pi) statistical and execution policy. The distinction between (O) and
(A) matters: JSON validity is executable but not generally a differentiable
training loss. Training may use reference sequences or teacher KL while final
acceptance executes an autoregressive JSON parser.

The notation (f_{\theta+\Delta_p}\models C) always means “passes under the
declared finite or sampled scope, generation policy, environment, and search
budget.” It is not a statement about all possible inputs.

## Patch compilation

The R1 artifact language admits low-rank matrix deltas, sparse residuals, and
vector deltas. The implemented automatic optimizer searches low-rank deltas on
selected `torch.nn.Linear` weights; sparse and vector operations are supported
by validation, mounting, and materialization but are not automatically
synthesized by that optimizer. Conceptually,

\[
\Delta_p^*=\arg\min_{\Delta\in\mathcal P}\mathcal C(\Delta)
\]

subject to target losses (L_t(\theta+\Delta)\le\tau_t) and preservation
distances

\[
D_j(f_{\theta+\Delta},f_\theta)\le\epsilon_j.
\]

The research objective can combine rank, active module count, sparse nonzeros,
Frobenius norm, and serialized bytes:

\[
\mathcal C(\Delta)=
\lambda_r\sum_l r_l+\lambda_m|\{l:r_l>0\}|+
\lambda_s|\Delta_{\mathrm{sparse}}|_0+
\lambda_n\|\Delta\|_F^2+\lambda_b\operatorname{bytes}(\Delta).
\]

The current optimizer enforces maximum-rank and maximum-module budgets and adds
a squared Frobenius penalty. Executed module/rank minimization is a separate
pass; sparse-count and serialized-byte terms are not yet direct optimizer terms.

### Candidate modules and contrastive bases

For candidate linear module (l), the implemented selection evidence contains
target and guard gradient norms, parameter count, and a dense-delta byte
estimate. A simple diagnostic is

\[
s_l=\frac{\|G_l^{\mathrm{target}}\|_F}
{\epsilon+\|G_l^{\mathrm{guard}}\|_F}.
\]

The implementation aggregates the declared objective and guard gradients. A
target-sensitive matrix is orthogonalized against the aggregate guard gradient,
and direct SVD is used below a configured element limit, with seeded randomized
SVD above it. It does not yet estimate per-example gradient covariances or use a
random-projection storage path. A target-delta initializer is available when a
matching target checkpoint exists,

\[
W_{\mathrm{target}}-W_{\mathrm{base}}
\]

but it is not automatically selected by the main compiler path. The complete
target delta is never implicitly declared to be the selected behavior.

### Constrained optimization

ModelPact uses a primal-dual augmented objective rather than silently replacing
constraints with a fixed weighted average:

\[
\mathcal L=\mathcal L_{\mathrm{target}}+
\sum_j\lambda_j\max(0,c_j(\Delta))+
\rho\sum_j\max(0,c_j(\Delta))^2+
\lambda_C\mathcal C(\Delta).
\]

Multipliers are nonnegative and updated from observed violations. The compiler
records target losses, guard violations, multipliers, gradient and patch norms,
active modules, ranks, and the best feasible step. Failure within its finite
step budget is empirical infeasibility within that budget, not mathematical
infeasibility.

### Counterexample-guided refinement

The CEGIS engine coordinates target and guard search callbacks, deduplicates
executed counterexamples, extends the working sets, and calls the compiler
again. The generic `compile` command derives a deterministic bounded mutation
space only from visible target/guard rows, executes every proposal it reports,
delta-debug minimizes supported failures, and recompiles exact/free-generation
targets plus base-KL guards. Unsupported assertion/search semantics produce an
explicit `UNSUPPORTED` result. Sealed holdout data is not part of the mutation
space and is opened only for the selected final candidate. “No counterexample
found” is always qualified by operators, prompt space, rounds, and execution
budget.

### Executed minimization

The implemented minimizer greedily tests individual module removals through a
caller-supplied verifier, then factorizes each retained dense matrix and tests
smaller ranks. It does not currently test module groups, briefly reoptimize
candidates, or prune sparse entries. Completed one-removal/rank loops can emit
`MODULE_ONE_MINIMAL` or `RANK_LOCAL_MINIMUM`; an incomplete search that exhausts
its verification allowance can emit `BUDGET_MINIMAL`. This path cannot support
`GLOBAL_MINIMUM`.

## Behavioral model diff and extraction

A difference witness is an input on which two executed local models diverge under
a declared metric. The CLI starts from fixed JSONL/string-list prompts or a
finite template grammar; the library also exposes a bounded Cartesian utility.
It applies deterministic
entity/number, casing, punctuation, order, role, distractor, synonym, whitespace,
and completion-prefix mutations. Witness objectives combine output divergence,
novelty, a token-flip coverage heuristic, and prompt complexity. Delta debugging
preserves the declared divergence threshold.

Each witness stores generated-output hashes, symmetric-KL/Jensen-Shannon and
top-token-flip-rate metrics, first generated-token diagnostics, projected
activation/gradient/prompt fingerprints, and provenance. It does not currently
store token-level flip locations. Deterministic clustering yields scoped
empirical groups with medoids, dispersion, uncertainty, and outliers. A cluster
is not a semantic ground-truth category.

Selective extraction makes the target model a teacher only on a selected witness
domain and makes the base model the teacher on nonselected clusters and supplied
controls. The extraction command compiles a low-rank candidate and executes
greedy free-generation equality checks for the selected and nonselected witness
sets. Broader controls and sealed holdouts exist only when they are explicitly
supplied by a benchmark/contract; the extraction routine does not infer them.
No production behavior name or fixture mapping participates.

## Additive algebra and failure of behavioral closure

For R1 additive patches,

\[
\Delta_{p\oplus q}=\Delta_p+\Delta_q.
\]

The mathematical tensor operation is commutative and associative. ModelPact does
not invent an order-sensitive weight algebra. The implementation nevertheless
uses patch-ID order for deterministic floating-point accumulation and does not
claim bitwise invariance under reassociation. The research question is whether
contract validity is closed under ordinary addition:

\[
\operatorname{Closed}_\theta(p,q)\iff
f_{\theta+\Delta_p+\Delta_q}\models C_p\cup C_q,
\]

including every declared guard and required free-generation assertion.

For signed contract margin (m_c), the behavioral interaction residual is

\[
I^c_{p,q}=m_c(\theta+\Delta_p+\Delta_q)-m_c(\theta+\Delta_p)
-m_c(\theta+\Delta_q)+m_c(\theta).
\]

Library helpers implement module/index overlap, low-rank principal angles,
generic tensor cosine similarity, output residuals, and contract-margin
interaction. The composition CLI executes the base, each singleton, and the
union, and writes module/sparse overlap, available row/column principal angles,
baseline/singleton margins, and pair contract interactions. Gradient,
activation, or raw-output diagnostics are marked `NOT_AVAILABLE` with a reason
when their prerequisite evidence was not collected. Executed contract outcomes
are authoritative.

## Semantic merge

When addition fails and contracts are not statically contradictory, the semantic
merge protocol can invoke a supplied joint compiler to produce a new candidate
delta:

\[
p\otimes_\theta q=\arg\min_r
\left[\mathcal C(\Delta_r)+\lambda_d
\|\Delta_r-(\Delta_p+\Delta_q)\|_2^2\right]
\]

subject to

\[
f_{\theta+\Delta_r}\models C_p\cup C_q.
\]

The library request pins the parent IDs/deltas, union contract IDs, summed
initial delta, module/base identities, and resource budget. The supplied backend
is responsible for resolving contract data and parent teachers and for returning
optimization evidence. ModelPactBench supplies a concrete constrained PyTorch
optimizer for its analytic merge organism. The `merge` CLI also supplies a
concrete joint compiler for the built-in tiny adapter: it initializes from the
summed parent delta, optimizes a low-rank residual against declared objectives,
parent-teacher objectives, and union guards, then executes the returned
candidate. Custom and Hugging Face adapters need an explicit trusted compiler
integration when repair is required and otherwise return `UNSUPPORTED`. The CLI
packages a successful selected candidate as a complete Behavior Patch Bundle
v1 with union contracts, copied resources, executed evidence, an independently
regenerated certificate, and standalone tools. Parameter averaging alone is
never called a semantic merge.

Exact-output or opposed-margin requirements on the same exact input can yield a
`STATIC_CONTRACT_CONTRADICTION` witness. A supplied compiler that exhausts its
declared budget without a candidate is reported as
`EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET` with its restarts, steps, and best
margins; other backend failures remain ordinary compiler/semantic failures.

## Higher-order composition audit

For (n) patches and inclusion vector (x\in\{0,1\}^n),

\[
f_{\theta,x}=f_{\theta+\sum_i x_i\Delta_i}.
\]

For every contract, the audit executes a signed margin (m_c(x)). Small pools
can execute all (2^n-1) nonempty subsets. Larger pools begin with the empty
stack, singletons, selected pairs, balanced random subsets, requested stacks,
and structurally high-risk sets.

A bounded sparse pseudo-Boolean surrogate guides—not decides—new executions:

\[
\hat m_c(x)=\beta_0+\sum_i\beta_i x_i+
\sum_{i<j}\beta_{ij}x_ix_j+
\sum_{i<j<k}\beta_{ijk}x_ix_jx_k.
\]

The feature expansion contains every lower-order term through the selected
degree. Elastic-net coordinate descent, deterministic cross-validation, and
bootstrap ensembles control complexity and estimate uncertainty. Acquisition
combines predicted negative margin, uncertainty, unexplored terms, novelty,
subset-size bounds, and dependencies. Every proposed subset is executed. A
failure is reduced by ddmin with real removal tests. Only execution of every
nonempty subset emits `EXHAUSTIVE_COMPOSITION_AUDIT`; a budgeted run with no
observed failure says `NO_FAILURE_FOUND_WITHIN_BUDGET`, while a budgeted run
that finds one says `FAILING_SUBSET_FOUND`.

## Semantic rebase

For source patch (p) on (\theta_0) and changed base (\theta_1), semantic
recompilation targets

\[
\rho_{\theta_0\rightarrow\theta_1}(p)=\arg\min_{p'}
D_T(f_{\theta_1+\Delta_{p'}},f_{\theta_0+\Delta_p})+
\lambda_GD_G(f_{\theta_1+\Delta_{p'}},f_{\theta_1})+
\lambda_C\mathcal C(\Delta_{p'}),
\]

subject to the original target and preservation contracts.

When schemas match, the rebase protocol first applies the original delta and
executes old targets, old guards, and new-base preservation. Only a passing
execution is `DIRECT_TRANSPLANT_VERIFIED`. Otherwise it can pass an old-patched
teacher and new-unpatched teacher to a supplied behavioral recompiler.
The `rebase` CLI implements that recompiler for built-in tiny-to-tiny transfers,
including compatible cross-architecture tiny models; successful candidates are
verified across every distinct bundled contract (including each sealed holdout
when configured) and packaged on the new base. Before either path, the CLI
executes the source patch's visible target and preservation assertions and
refuses to distill a failing teacher. Original guards are retargeted relative to
the new unpatched base; `--new-base-policy` can add an explicit guard-only
control contract. Semantic tiny-model results undergo bounded executed module
and rank minimization when their state is matrix-valued. Custom and Hugging Face
adapters can use the verified direct path, but semantic recompilation requires
an explicit trusted compiler integration.
ModelPactBench also executes same-family and controlled cross-architecture
analytic cases. Cross-architecture rebase requires compatible tokenizer and
output semantics and never maps physical tensors.

## Revert and evidence semantics

Runtime unmount is exact because base parameters were never changed. Selecting
the original pinned checkpoint restores its recorded base identity; the revert
command does not recreate missing checkpoint bytes. Subtracting a folded
floating-point delta is only a numeric inverse. Removing a patch from a verified
stack can invalidate remaining contracts, so the CLI reconstructs and
re-verifies the remaining additive stack. A passing nonempty reconstruction is
`VERIFIED_LOGICAL_STACK_RECONSTRUCTED`; it is not runtime unmount, base-hash
restoration, numeric inversion, or semantic recompilation. The empty remainder
is `BASE_HASH_RESTORED` only because resolution selects the original pinned base.
Stack resolution executes the requested bounded subset audit before selecting a
final candidate, pins the audit and certificate hashes in the lockfile, and
emits a complete resolved patch bundle. When `repair_conflicts = true`, the
built-in tiny adapter invokes the real semantic-merge compiler; other adapters
return `UNSUPPORTED` when repair is actually required.

Certificates are claims about an execution envelope, not self-authenticating
proofs. Independent verification re-hashes data and reruns assertions through a
caller-supplied model-backed provider; the caller is responsible for independently
applying/reconstructing the patch. The strongest permitted sentence remains:

> Verified under the declared contracts, probe spaces, generation policy,
> environment, and search budget.

## Limitations

R1 is restricted to trusted local PyTorch adapters, decoder-only causal language
models, additive patches, and SafeTensors. The committed automated evidence is
CPU-only; device selection exists, but no GPU compatibility/performance claim is
made without a recorded run. Checkpoint materialization preflights metadata,
aliases, and the program, then processes one deterministic output shard at a
time. This bounds checkpoint-state residency but is not constant memory: patch
factors, a target delta, a shard, the SafeTensors snapshot copy, and an oversized
single tensor can dominate. The manifest records the exact configured/planned
sizes and measured I/O; peak RSS is the platform's process-lifetime high-water
mark or explicitly unavailable, not an isolated causal memory measurement. It
does not cover
remote/closed models, multimodal or diffusion models, RL preference training,
quantized training, arbitrary
distributed/tensor parallel checkpoints, MoE, routing, signing, a registry,
unlearning, arbitrary architecture translation, or formal verification of
open-ended behavior. Finite tests can miss failures; optimization can fail;
compact patches may not exist; interaction surrogates can miss subsets; and a
patch that passes on one software/hardware environment may be only numerically
or statistically reproducible elsewhere.
