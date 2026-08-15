"""Executed exhaustive and active-search audits over patch combinations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from modelpact.audit.design import InitialDesignConfig, initial_design
from modelpact.audit.reduce import (
    ReductionBudgetExhausted,
    ReductionResult,
    ddmin_failing_subset,
)
from modelpact.audit.subsets import (
    PatchSubset,
    canonical_subset,
    enumerate_subsets,
    hamming_distance,
    subset_to_vector,
    validate_patch_universe,
)
from modelpact.audit.surrogate import (
    BootstrapEnsemble,
    deterministic_cross_validate,
    fit_bootstrap_ensemble,
    hierarchical_feature_terms,
)
from modelpact.status import AuditClaim, CompositionCoverage, VerificationOutcome


@dataclass(frozen=True, slots=True)
class SubsetEvaluation:
    subset: PatchSubset
    margins: Mapping[str, float]
    outcome: VerificationOutcome
    violated_contracts: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.margins.values()):
            raise ValueError("contract margins must be finite")
        if self.subset and self.outcome is VerificationOutcome.PASS and not self.margins:
            raise ValueError("a passing nonempty subset must report executed contract margins")
        if self.outcome is VerificationOutcome.PASS and any(
            value < 0.0 for value in self.margins.values()
        ):
            raise ValueError("PASS subset evaluation cannot contain a negative margin")

    @property
    def passed(self) -> bool:
        return self.outcome is VerificationOutcome.PASS and all(
            value >= 0.0 for value in self.margins.values()
        )


class SubsetOracle(Protocol):
    def __call__(self, subset: PatchSubset) -> SubsetEvaluation: ...


@dataclass(frozen=True, slots=True)
class AuditConfig:
    subset_budget: int
    maximum_order: int | None = None
    exhaustive_threshold: int = 6
    surrogate_degree: int = 3
    include_all_pairs: bool = True
    initial_random_subsets: int = 0
    active_batch_size: int = 1
    bootstrap_samples: int = 24
    regularization_alphas: tuple[float, ...] = (0.001, 0.01, 0.1)
    l1_ratio: float = 0.8
    cross_validation_folds: int = 4
    seed: int = 0
    evaluate_empty_stack: bool = True
    reduce_failures: bool = True

    def __post_init__(self) -> None:
        if self.subset_budget <= 0:
            raise ValueError("subset_budget must be positive")
        if self.exhaustive_threshold < 0:
            raise ValueError("exhaustive_threshold must be non-negative")
        if self.surrogate_degree not in (1, 2, 3):
            raise ValueError("surrogate_degree must be one, two, or three")
        if self.active_batch_size <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("active batch and bootstrap sample counts must be positive")
        if self.initial_random_subsets < 0:
            raise ValueError("initial_random_subsets must be non-negative")
        if not self.regularization_alphas or any(alpha < 0 for alpha in self.regularization_alphas):
            raise ValueError("regularization_alphas must be non-empty and non-negative")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least two")


@dataclass(frozen=True, slots=True)
class SurrogateFitEvidence:
    contract_id: str
    observations: int
    selected_alpha: float
    cross_validation_mse: float | None
    degree: int


@dataclass(frozen=True, slots=True)
class ActiveProposal:
    subset: PatchSubset
    score: float
    predicted_worst_margin: float | None
    predictive_uncertainty: float
    novelty: float
    unexplored_interaction_fraction: float


@dataclass(frozen=True, slots=True)
class AuditResult:
    patch_ids: tuple[str, ...]
    possible_nonempty_subsets: int
    subset_budget: int
    baseline: SubsetEvaluation | None
    evaluations: tuple[SubsetEvaluation, ...]
    claims: tuple[AuditClaim, ...]
    coverage: CompositionCoverage
    failing_subsets: tuple[PatchSubset, ...]
    minimal_failures: tuple[ReductionResult, ...]
    reduction_attempts: tuple[ReductionResult, ...]
    active_proposals: tuple[ActiveProposal, ...]
    surrogate_fits: tuple[SurrogateFitEvidence, ...]
    search_space_exhausted: bool
    budget_exhausted: bool

    @property
    def executed_subset_count(self) -> int:
        return len(self.evaluations)

    @property
    def total_model_executions(self) -> int:
        return len(self.evaluations) + (1 if self.baseline is not None else 0)

    @property
    def all_combinations_passed(self) -> bool:
        return self.search_space_exhausted and not self.failing_subsets


def _dependencies_satisfied(subset: PatchSubset, dependencies: Mapping[str, Sequence[str]]) -> bool:
    members = set(subset)
    return all(set(dependencies.get(patch, ())) <= members for patch in subset)


def _fit_surrogates(
    *,
    patch_ids: tuple[str, ...],
    evaluations: Sequence[SubsetEvaluation],
    config: AuditConfig,
) -> tuple[dict[str, BootstrapEnsemble], tuple[SurrogateFitEvidence, ...]]:
    contract_ids = sorted(
        {contract for evaluation in evaluations for contract in evaluation.margins}
    )
    models: dict[str, BootstrapEnsemble] = {}
    evidence: list[SurrogateFitEvidence] = []
    for offset, contract_id in enumerate(contract_ids):
        relevant = [evaluation for evaluation in evaluations if contract_id in evaluation.margins]
        vectors = [subset_to_vector(evaluation.subset, patch_ids) for evaluation in relevant]
        margins = [evaluation.margins[contract_id] for evaluation in relevant]
        if not vectors:
            continue
        if len(vectors) >= 2:
            cross_validation = deterministic_cross_validate(
                vectors,
                margins,
                patch_count=len(patch_ids),
                degree=config.surrogate_degree,
                alphas=config.regularization_alphas,
                l1_ratio=config.l1_ratio,
                folds=config.cross_validation_folds,
                seed=config.seed + offset,
            )
            alpha = cross_validation.alpha
            cross_validation_mse = cross_validation.mean_squared_error
        else:
            alpha = min(config.regularization_alphas)
            cross_validation_mse = None
        models[contract_id] = fit_bootstrap_ensemble(
            vectors,
            margins,
            patch_count=len(patch_ids),
            degree=config.surrogate_degree,
            alpha=alpha,
            l1_ratio=config.l1_ratio,
            samples=config.bootstrap_samples,
            seed=config.seed + 10_007 * (offset + 1),
        )
        evidence.append(
            SurrogateFitEvidence(
                contract_id=contract_id,
                observations=len(vectors),
                selected_alpha=alpha,
                cross_validation_mse=cross_validation_mse,
                degree=config.surrogate_degree,
            )
        )
    return models, tuple(evidence)


def select_active_subsets(
    *,
    patch_ids: Sequence[str],
    candidates: Sequence[PatchSubset],
    executed: Sequence[PatchSubset],
    models: Mapping[str, BootstrapEnsemble],
    count: int,
    degree: int,
    dependencies: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ActiveProposal, ...]:
    """Rank candidates; every returned subset still requires oracle execution."""

    if count <= 0:
        return ()
    universe = validate_patch_universe(patch_ids)
    dependency_map = dependencies or {}
    executed_vectors = [subset_to_vector(subset, universe) for subset in executed]
    terms = hierarchical_feature_terms(len(universe), degree=degree)
    explored_terms = {
        term
        for term in terms
        if any(all(vector[index] for index in term) for vector in executed_vectors)
    }
    proposals: list[ActiveProposal] = []
    for subset in candidates:
        canonical = canonical_subset(subset, universe=universe)
        if not _dependencies_satisfied(canonical, dependency_map):
            continue
        vector = subset_to_vector(canonical, universe)
        predictions = [model.predict_one(vector) for model in models.values()]
        predicted_worst = min((prediction.mean for prediction in predictions), default=None)
        uncertainty = max(
            (prediction.standard_deviation for prediction in predictions), default=0.0
        )
        if executed_vectors:
            novelty = min(
                hamming_distance(vector, prior) / len(universe) for prior in executed_vectors
            )
        else:
            novelty = 1.0
        active_terms = [term for term in terms if all(vector[index] for index in term)]
        unexplored = sum(term not in explored_terms for term in active_terms)
        unexplored_fraction = unexplored / len(active_terms) if active_terms else 0.0
        # Negative predicted margins, uncertainty, unobserved interaction terms,
        # and design novelty all increase priority.  This score chooses inputs;
        # it is never interpreted as an acceptance result.
        predicted_risk = max(0.0, -(predicted_worst or 0.0))
        score = predicted_risk + uncertainty + unexplored_fraction + 0.25 * novelty
        proposals.append(
            ActiveProposal(
                subset=canonical,
                score=score,
                predicted_worst_margin=predicted_worst,
                predictive_uncertainty=uncertainty,
                novelty=novelty,
                unexplored_interaction_fraction=unexplored_fraction,
            )
        )
    proposals.sort(key=lambda proposal: (-proposal.score, len(proposal.subset), proposal.subset))
    return tuple(proposals[:count])


def audit_patch_pool(
    patch_ids: Sequence[str],
    *,
    oracle: SubsetOracle,
    config: AuditConfig,
    dependencies: Mapping[str, Sequence[str]] | None = None,
    user_requested: Sequence[Sequence[str]] = (),
    high_risk: Sequence[Sequence[str]] = (),
) -> AuditResult:
    """Execute a complete small audit or a bounded active subset search.

    ``subset_budget`` counts nonempty patch combinations.  The optional empty
    baseline is recorded separately, so a six-patch exhaustive audit contains 63
    nonempty evaluations (and, when enabled, one baseline execution).
    """

    universe = validate_patch_universe(patch_ids)
    if config.subset_budget < len(universe):
        raise ValueError("subset_budget must permit verification of every singleton")
    maximum_order = config.maximum_order or len(universe)
    if maximum_order < 1 or maximum_order > len(universe):
        raise ValueError("maximum_order must be between one and the patch count")
    possible_count = (1 << len(universe)) - 1
    all_nonempty = enumerate_subsets(universe)
    eligible = enumerate_subsets(universe, maximum_order=maximum_order)
    baseline: SubsetEvaluation | None = None
    if config.evaluate_empty_stack:
        baseline = oracle(())
        if baseline.subset != ():
            raise ValueError("empty-stack oracle evaluation reported a different subset")

    evaluations: dict[PatchSubset, SubsetEvaluation] = {}
    execution_order: list[PatchSubset] = []

    def execute(subset: PatchSubset) -> SubsetEvaluation:
        canonical = canonical_subset(subset, universe=universe)
        if not canonical:
            if baseline is None:
                raise ValueError("empty-stack evaluation was not requested")
            return baseline
        if canonical in evaluations:
            return evaluations[canonical]
        if len(evaluations) >= config.subset_budget:
            raise ReductionBudgetExhausted
        result = oracle(canonical)
        if canonical_subset(result.subset, universe=universe) != canonical:
            raise ValueError("subset oracle reported an identity different from its request")
        evaluations[canonical] = result
        execution_order.append(canonical)
        return result

    exhaustive_requested = (
        len(universe) <= config.exhaustive_threshold
        and maximum_order == len(universe)
        and config.subset_budget >= possible_count
    )
    proposal_evidence: list[ActiveProposal] = []
    latest_fit_evidence: tuple[SurrogateFitEvidence, ...] = ()
    if exhaustive_requested:
        for subset in all_nonempty:
            execute(subset)
    else:
        design = initial_design(
            universe,
            config=InitialDesignConfig(
                include_pairs=config.include_all_pairs,
                balanced_random_subsets=config.initial_random_subsets,
                maximum_order=maximum_order,
                seed=config.seed,
            ),
            user_requested=user_requested,
            high_risk=high_risk,
        )
        for subset in design:
            if len(evaluations) >= config.subset_budget:
                break
            execute(subset)
        while len(evaluations) < config.subset_budget:
            remaining = [subset for subset in eligible if subset not in evaluations]
            if not remaining:
                break
            models, latest_fit_evidence = _fit_surrogates(
                patch_ids=universe,
                evaluations=[evaluations[subset] for subset in execution_order],
                config=config,
            )
            proposals = select_active_subsets(
                patch_ids=universe,
                candidates=remaining,
                executed=execution_order,
                models=models,
                count=min(config.active_batch_size, config.subset_budget - len(evaluations)),
                degree=config.surrogate_degree,
                dependencies=dependencies,
            )
            if not proposals:
                break
            for proposal in proposals:
                execute(proposal.subset)
                proposal_evidence.append(proposal)

    initially_failing = tuple(
        subset for subset in execution_order if not evaluations[subset].passed
    )
    reductions: list[ReductionResult] = []
    reduction_attempts: list[ReductionResult] = []
    seen_reduced: set[PatchSubset] = set()
    if config.reduce_failures:
        for subset in sorted(initially_failing, key=lambda item: (len(item), item)):
            reduction = ddmin_failing_subset(
                subset,
                oracle=lambda candidate: not execute(candidate).passed,
                initial_known_failing=True,
            )
            reduction_attempts.append(reduction)
            if reduction.one_minimal and reduction.reduced not in seen_reduced:
                reductions.append(reduction)
                seen_reduced.add(reduction.reduced)

    # Reduction is itself executed verification and can discover additional
    # failing complements.  Preserve every such outcome in the audit record.
    failing = tuple(subset for subset in execution_order if not evaluations[subset].passed)
    search_space_exhausted = set(evaluations) == set(all_nonempty)
    budget_exhausted = len(evaluations) >= config.subset_budget and not search_space_exhausted
    claims: list[AuditClaim] = []
    if search_space_exhausted:
        claims.append(AuditClaim.EXHAUSTIVE_COMPOSITION_AUDIT)
        coverage = CompositionCoverage.EXHAUSTIVE_SUBSETS
    else:
        claims.append(AuditClaim.BUDGETED_COMPOSITION_AUDIT)
        coverage = CompositionCoverage.ACTIVE_BUDGETED
    if failing:
        claims.append(AuditClaim.FAILING_SUBSET_FOUND)
    elif not search_space_exhausted:
        claims.append(AuditClaim.NO_FAILURE_FOUND_WITHIN_BUDGET)
    return AuditResult(
        patch_ids=universe,
        possible_nonempty_subsets=possible_count,
        subset_budget=config.subset_budget,
        baseline=baseline,
        evaluations=tuple(evaluations[subset] for subset in execution_order),
        claims=tuple(claims),
        coverage=coverage,
        failing_subsets=failing,
        minimal_failures=tuple(reductions),
        reduction_attempts=tuple(reduction_attempts),
        active_proposals=tuple(proposal_evidence),
        surrogate_fits=latest_fit_evidence,
        search_space_exhausted=search_space_exhausted,
        budget_exhausted=budget_exhausted,
    )
