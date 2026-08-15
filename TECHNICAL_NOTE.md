# ModelPact: Contract-Carrying Learned Behavior Patches

## Abstract

ModelPact studies whether a learned behavioral change can be distributed as a
scoped software artifact rather than as an opaque checkpoint or parameter delta.
The artifact combines an additive low-rank-plus-sparse program with executable
target and preservation contracts, lineage, and independently regenerable
evidence. The system does not infer complete model semantics and does not certify
open-ended safety. It executes finite, sampled, and searched tests whose scope is
recorded explicitly.

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

R1 searches a program family containing low-rank matrix deltas, sparse residuals,
and vector deltas. Conceptually,

\[
\Delta_p^*=\arg\min_{\Delta\in\mathcal P}\mathcal C(\Delta)
\]

subject to target losses (L_t(\theta+\Delta)\le\tau_t) and preservation
distances

\[
D_j(f_{\theta+\Delta},f_\theta)\le\epsilon_j.
\]

The complexity objective can combine rank, active module count, sparse nonzeros,
Frobenius norm, and serialized bytes:

\[
\mathcal C(\Delta)=
\lambda_r\sum_l r_l+\lambda_m|\{l:r_l>0\}|+
\lambda_s|\Delta_{\mathrm{sparse}}|_0+
\lambda_n\|\Delta\|_F^2+\lambda_b\operatorname{bytes}(\Delta).
\]

### Candidate modules and contrastive bases

For candidate module (l), the initial evidence includes target and guard
gradient norms, activation differences, physical target deltas when available,
module size, and runtime cost. A simple diagnostic is

\[
s_l=\frac{\|G_l^{\mathrm{target}}\|_F}
{\epsilon+\|G_l^{\mathrm{guard}}\|_F}.
\]

The implementation accumulates gradients in microbatches and can project them
before storage. A target-sensitive direction is orthogonalized against an
aggregate guard gradient, and bounded direct or randomized SVD supplies an
initial low-rank basis. When a matching target checkpoint exists,

\[
W_{\mathrm{target}}-W_{\mathrm{base}}
\]

is only an initialization signal. The complete target delta is never implicitly
declared to be the selected behavior.

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
records losses, margins, multipliers, gradient and patch norms, active modules,
ranks, and the best feasible candidate. Failure within a finite restart/step
budget is empirical infeasibility within that budget, not mathematical
infeasibility.

### Counterexample-guided refinement

After producing a candidate, deterministic local fuzzing searches target domains
for lost margins, teacher divergence, format failures, and brittle prompt
variants. Guard neighborhoods are searched for patched/base divergence,
assertion flips, neighboring behavior drift, and unstable generation. A found
failure is minimized by clause/token delta debugging, added to the working set,
and the patch is recompiled. “No counterexample found” is always qualified by
operators, prompt space, rounds, and execution budget.

### Executed minimization

Module groups and individual modules are removed and all contracts rerun. Each
effective matrix delta is factorized and smaller ranks are tested, optionally
with brief reoptimization. Sparse entries or blocks are pruned only after an
executed candidate passes. Greedy one-removal and local rank searches can support
`MODULE_ONE_MINIMAL`, `RANK_LOCAL_MINIMUM`, or `BUDGET_MINIMAL`; they cannot
support `GLOBAL_MINIMUM`.

## Behavioral model diff and extraction

A difference witness is an input on which two executed local models diverge under
a declared metric. ModelPact starts from fixed JSONL prompts, finite grammars,
Cartesian generators, or trusted local generators, and applies deterministic
entity/number, casing, punctuation, order, role, distractor, synonym, whitespace,
and completion-prefix mutations. Witness objectives combine output divergence,
novelty, cluster coverage, validity, and prompt complexity. Delta debugging
preserves the declared divergence threshold.

Each witness stores generated-output hashes, token flips and divergences,
projected activation and gradient differences, and provenance. Deterministic
clustering yields scoped empirical groups with medoids, dispersion, uncertainty,
and outliers. A cluster is not a semantic ground-truth category.

Selective extraction makes the target model a teacher only on a selected witness
domain and makes the base model the teacher on nonselected clusters and broad
controls. It then runs the ordinary compiler and free-generation verification.
This is the mechanism that rejects unrelated physical changes; no production
behavior name or fixture mapping participates.

## Additive algebra and failure of behavioral closure

For R1 additive patches,

\[
\Delta_{p\oplus q}=\Delta_p+\Delta_q.
\]

The tensor operation is commutative and associative. ModelPact does not invent an
order-sensitive weight algebra. The research question is whether contract
validity is closed under this ordinary addition:

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

Module/index overlap, principal angles, gradient cosine, and activation-delta
similarity are diagnostics. Executed contract outcomes are authoritative.

## Semantic merge

When addition fails and contracts are not statically contradictory, ModelPact
jointly recompiles a new patch:

\[
p\otimes_\theta q=\arg\min_r
\left[\mathcal C(\Delta_r)+\lambda_d
\|\Delta_r-(\Delta_p+\Delta_q)\|_2^2\right]
\]

subject to

\[
f_{\theta+\Delta_r}\models C_p\cup C_q.
\]

The union specification contains all parent targets and guards, base
preservation, and parent-patched behavior teachers on each parent domain.
Initialization can concatenate parent factors or start from their sum. Parent
multipliers and optional conflict-aware gradient projection remain diagnostics
inside an actual new optimization. Parameter averaging alone is never called a
semantic merge.

Exact-output or opposed-margin requirements on the same exact input can yield a
minimal `STATIC_CONTRACT_CONTRADICTION` witness. Otherwise a failed search is
`EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET` with restarts, steps, and best margins.

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

Hierarchical feature inclusion, elastic-net/L1 regularization, deterministic
cross-validation, and bootstrap ensembles control complexity and estimate
uncertainty. Acquisition combines predicted negative margin, uncertainty,
unexplored terms, novelty, subset-size bounds, and dependencies. Every proposed
subset is executed. A failure is reduced by ddmin with real removal tests.
Only full enumeration emits `EXHAUSTIVE_COMPOSITION_AUDIT`; all other negative
findings say `NO_FAILURE_FOUND_WITHIN_BUDGET`.

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

When schemas match, the system first applies the original delta and executes
old targets, old guards, and new-base preservation. Only a passing execution is
`DIRECT_TRANSPLANT_VERIFIED`. A failure starts compilation with the old patched
model as target teacher and the new unpatched model as preservation teacher.
Cross-architecture tiny rebase requires compatible tokenizer and output
semantics and never maps physical tensors.

## Revert and evidence semantics

Runtime unmount is exact because base parameters were never changed. Restoring
the original available checkpoint by hash is also exact. Subtracting a folded
floating-point delta is only a numeric inverse. Removing a patch from a verified
stack can invalidate remaining contracts, so the resolver reconstructs,
re-verifies, and if authorized recompiles the remaining semantic stack.

Certificates are claims about an execution envelope, not self-authenticating
proofs. Independent verification re-hashes data, reconstructs the patched model,
and reruns assertions. The strongest permitted sentence remains:

> Verified under the declared contracts, probe spaces, generation policy,
> environment, and search budget.

## Limitations

R1 is restricted to trusted local PyTorch adapters, decoder-only causal language
models, additive patches, SafeTensors, CPU verification, and practical
single-NVIDIA-GPU compilation. It does not cover remote/closed models, multimodal
or diffusion models, RL preference training, quantized training, arbitrary
distributed/tensor parallel checkpoints, MoE, routing, signing, a registry,
unlearning, arbitrary architecture translation, or formal verification of
open-ended behavior. Finite tests can miss failures; optimization can fail;
compact patches may not exist; interaction surrogates can miss subsets; and a
patch that passes on one software/hardware environment may be only numerically
or statistically reproducible elsewhere.

