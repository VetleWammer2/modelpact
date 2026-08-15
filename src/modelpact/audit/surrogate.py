"""Sparse pseudo-Boolean interaction surrogates used only for audit search."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from modelpact.audit.subsets import InclusionVector

FeatureTerm = tuple[int, ...]


def hierarchical_feature_terms(patch_count: int, *, degree: int = 2) -> tuple[FeatureTerm, ...]:
    """Return all lower-order terms required by interactions through ``degree``."""

    if patch_count < 1:
        raise ValueError("patch_count must be positive")
    if degree not in (1, 2, 3):
        raise ValueError("pseudo-Boolean degree must be one, two, or three")
    return tuple(
        term
        for order in range(1, degree + 1)
        for term in combinations(range(patch_count), order)
    )


def validate_vectors(vectors: Sequence[InclusionVector], *, patch_count: int) -> None:
    if not vectors:
        raise ValueError("at least one observed subset is required")
    for vector in vectors:
        if len(vector) != patch_count:
            raise ValueError("inclusion vector length does not match patch_count")
        if any(value not in (0, 1) for value in vector):
            raise ValueError("inclusion vectors must be binary")


def feature_value(vector: InclusionVector, term: FeatureTerm) -> float:
    return float(all(vector[index] == 1 for index in term))


def pseudo_boolean_features(
    vector: InclusionVector, terms: Sequence[FeatureTerm]
) -> tuple[float, ...]:
    return tuple(feature_value(vector, term) for term in terms)


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


class PseudoBooleanRegressor:
    """Deterministic elastic-net coordinate descent for bounded subset audits."""

    def __init__(
        self,
        patch_count: int,
        *,
        degree: int = 2,
        alpha: float = 0.01,
        l1_ratio: float = 0.8,
        maximum_iterations: int = 2_000,
        tolerance: float = 1e-9,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        if maximum_iterations <= 0 or tolerance <= 0:
            raise ValueError("iteration count and tolerance must be positive")
        self.patch_count = patch_count
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.terms = hierarchical_feature_terms(patch_count, degree=degree)
        self.intercept_: float | None = None
        self.coefficients_: tuple[float, ...] | None = None
        self.iterations_: int = 0

    def fit(
        self, vectors: Sequence[InclusionVector], margins: Sequence[float]
    ) -> PseudoBooleanRegressor:
        validate_vectors(vectors, patch_count=self.patch_count)
        if len(vectors) != len(margins):
            raise ValueError("vectors and margins must contain the same number of observations")
        if not all(math.isfinite(value) for value in margins):
            raise ValueError("surrogate margins must be finite")
        row_count = len(vectors)
        matrix = [list(pseudo_boolean_features(vector, self.terms)) for vector in vectors]
        column_means = [
            sum(matrix[row][column] for row in range(row_count)) / row_count
            for column in range(len(self.terms))
        ]
        centered = [
            [matrix[row][column] - column_means[column] for column in range(len(self.terms))]
            for row in range(row_count)
        ]
        target_mean = sum(margins) / row_count
        residual = [value - target_mean for value in margins]
        coefficients = [0.0] * len(self.terms)
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        for iteration in range(1, self.maximum_iterations + 1):
            largest_change = 0.0
            for column in range(len(self.terms)):
                old = coefficients[column]
                column_values = [centered[row][column] for row in range(row_count)]
                squared_norm = sum(value * value for value in column_values) / row_count
                denominator = squared_norm + l2_penalty
                if denominator == 0.0:
                    new = 0.0
                else:
                    partial = sum(
                        column_values[row] * (residual[row] + column_values[row] * old)
                        for row in range(row_count)
                    ) / row_count
                    new = _soft_threshold(partial, l1_penalty) / denominator
                if new != old:
                    for row, value in enumerate(column_values):
                        residual[row] += value * (old - new)
                    coefficients[column] = new
                    largest_change = max(largest_change, abs(new - old))
            self.iterations_ = iteration
            if largest_change <= self.tolerance:
                break
        self.intercept_ = target_mean - sum(
            coefficient * mean for coefficient, mean in zip(coefficients, column_means, strict=True)
        )
        self.coefficients_ = tuple(coefficients)
        return self

    def predict_one(self, vector: InclusionVector) -> float:
        if self.intercept_ is None or self.coefficients_ is None:
            raise RuntimeError("surrogate has not been fitted")
        if len(vector) != self.patch_count or any(value not in (0, 1) for value in vector):
            raise ValueError("invalid inclusion vector")
        features = pseudo_boolean_features(vector, self.terms)
        return self.intercept_ + sum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients_, features, strict=True)
        )

    def predict(self, vectors: Sequence[InclusionVector]) -> tuple[float, ...]:
        return tuple(self.predict_one(vector) for vector in vectors)

    def coefficient_map(self) -> dict[FeatureTerm, float]:
        if self.coefficients_ is None:
            raise RuntimeError("surrogate has not been fitted")
        return dict(zip(self.terms, self.coefficients_, strict=True))


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    alpha: float
    mean_squared_error: float | None
    fold_errors: tuple[float, ...]


def deterministic_cross_validate(
    vectors: Sequence[InclusionVector],
    margins: Sequence[float],
    *,
    patch_count: int,
    degree: int,
    alphas: Sequence[float],
    l1_ratio: float,
    folds: int,
    seed: int,
) -> CrossValidationResult:
    validate_vectors(vectors, patch_count=patch_count)
    if len(vectors) != len(margins):
        raise ValueError("vectors and margins must contain the same number of observations")
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("alphas must contain non-negative candidates")
    if folds < 2:
        raise ValueError("cross-validation needs at least two folds")
    actual_folds = min(folds, len(vectors))
    if actual_folds < 2:
        # A single observation has no honest held-out estimate.
        return CrossValidationResult(float(alphas[0]), None, ())
    indices = list(range(len(vectors)))
    random.Random(seed).shuffle(indices)  # noqa: S311 -- deterministic fold assignment
    assignments = {index: position % actual_folds for position, index in enumerate(indices)}
    results: list[CrossValidationResult] = []
    for alpha in sorted({float(value) for value in alphas}):
        errors: list[float] = []
        for fold in range(actual_folds):
            train = [index for index in range(len(vectors)) if assignments[index] != fold]
            test = [index for index in range(len(vectors)) if assignments[index] == fold]
            model = PseudoBooleanRegressor(
                patch_count,
                degree=degree,
                alpha=alpha,
                l1_ratio=l1_ratio,
            ).fit([vectors[index] for index in train], [margins[index] for index in train])
            predictions = model.predict([vectors[index] for index in test])
            errors.append(
                sum(
                    (prediction - margins[index]) ** 2
                    for prediction, index in zip(predictions, test, strict=True)
                )
                / len(test)
            )
        results.append(CrossValidationResult(alpha, sum(errors) / len(errors), tuple(errors)))
    return min(
        results,
        key=lambda result: (
            result.mean_squared_error
            if result.mean_squared_error is not None
            else math.inf,
            result.alpha,
        ),
    )


@dataclass(frozen=True, slots=True)
class UncertainPrediction:
    mean: float
    standard_deviation: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class BootstrapEnsemble:
    models: tuple[PseudoBooleanRegressor, ...]
    confidence_level: float

    def predict_one(self, vector: InclusionVector) -> UncertainPrediction:
        if not self.models:
            raise RuntimeError("bootstrap ensemble is empty")
        values = sorted(model.predict_one(vector) for model in self.models)
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        tail = (1.0 - self.confidence_level) / 2.0

        def quantile(probability: float) -> float:
            if len(values) == 1:
                return values[0]
            position = probability * (len(values) - 1)
            lower_index = math.floor(position)
            upper_index = math.ceil(position)
            fraction = position - lower_index
            return values[lower_index] * (1.0 - fraction) + values[upper_index] * fraction

        return UncertainPrediction(
            mean=mean,
            standard_deviation=math.sqrt(variance),
            lower=quantile(tail),
            upper=quantile(1.0 - tail),
        )


def fit_bootstrap_ensemble(
    vectors: Sequence[InclusionVector],
    margins: Sequence[float],
    *,
    patch_count: int,
    degree: int,
    alpha: float,
    l1_ratio: float,
    samples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> BootstrapEnsemble:
    validate_vectors(vectors, patch_count=patch_count)
    if len(vectors) != len(margins):
        raise ValueError("vectors and margins must contain the same number of observations")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    generator = random.Random(seed)  # noqa: S311 -- deterministic statistical bootstrap
    models: list[PseudoBooleanRegressor] = []
    for _ in range(samples):
        indices = [generator.randrange(len(vectors)) for _ in vectors]
        model = PseudoBooleanRegressor(
            patch_count,
            degree=degree,
            alpha=alpha,
            l1_ratio=l1_ratio,
        ).fit([vectors[index] for index in indices], [margins[index] for index in indices])
        models.append(model)
    return BootstrapEnsemble(tuple(models), confidence_level)
