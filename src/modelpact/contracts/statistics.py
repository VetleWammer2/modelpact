"""Deterministic statistical utilities used by verification and benchmarks."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    samples: int
    seed: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "samples": self.samples,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    interval: ConfidenceInterval
    probability_greater_than_zero: float
    pairs: int


def _validate_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _validate_policy(confidence_level: float, samples: int, seed: int) -> None:
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    if isinstance(samples, bool) or not 1 <= samples <= 1_000_000:
        raise ValueError("samples must be in [1, 1000000]")
    if isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative signed 64-bit integer")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if probability <= 0.0:
        return float(sorted_values[0])
    if probability >= 1.0:
        return float(sorted_values[-1])
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    return lower + fraction * (upper - lower)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    samples: int = 2000,
    seed: int = 81273,
) -> ConfidenceInterval:
    observed = _validate_values(values, "values")
    _validate_policy(confidence_level, samples, seed)
    # Statistical resampling needs reproducibility, not cryptographic randomness.
    generator = random.Random(seed)  # noqa: S311
    count = len(observed)
    estimates = [
        fmean(observed[generator.randrange(count)] for _ in range(count)) for _ in range(samples)
    ]
    estimates.sort()
    alpha = (1.0 - confidence_level) / 2.0
    return ConfidenceInterval(
        estimate=fmean(observed),
        lower=_percentile(estimates, alpha),
        upper=_percentile(estimates, 1.0 - alpha),
        confidence_level=confidence_level,
        method="percentile_bootstrap_mean",
        samples=samples,
        seed=seed,
    )


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence_level: float = 0.95,
    samples: int = 2000,
    seed: int = 81273,
) -> PairedBootstrapResult:
    """Bootstrap the mean paired difference ``left - right``.

    Pair indices, rather than each arm independently, are resampled.  This
    preserves prompt/model-seed pairing and is deterministic for a fixed seed.
    """

    left_values = _validate_values(left, "left")
    right_values = _validate_values(right, "right")
    if len(left_values) != len(right_values):
        raise ValueError("paired bootstrap inputs must have equal length")
    _validate_policy(confidence_level, samples, seed)
    differences = tuple(a - b for a, b in zip(left_values, right_values, strict=True))
    # Statistical resampling needs reproducibility, not cryptographic randomness.
    generator = random.Random(seed)  # noqa: S311
    count = len(differences)
    estimates = [
        fmean(differences[generator.randrange(count)] for _ in range(count)) for _ in range(samples)
    ]
    positive = sum(value > 0.0 for value in estimates) / samples
    estimates.sort()
    alpha = (1.0 - confidence_level) / 2.0
    interval = ConfidenceInterval(
        estimate=fmean(differences),
        lower=_percentile(estimates, alpha),
        upper=_percentile(estimates, 1.0 - alpha),
        confidence_level=confidence_level,
        method="paired_percentile_bootstrap_mean_difference",
        samples=samples,
        seed=seed,
    )
    return PairedBootstrapResult(
        interval=interval,
        probability_greater_than_zero=positive,
        pairs=count,
    )


def adjust_p_values(values: Sequence[float], *, method: str) -> tuple[float, ...]:
    """Apply a deterministic Bonferroni or Holm family-wise correction."""

    p_values = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be finite and in [0, 1]")
    if method == "none":
        return p_values
    count = len(p_values)
    if method == "bonferroni":
        return tuple(min(1.0, value * count) for value in p_values)
    if method != "holm":
        raise ValueError("method must be none, holm, or bonferroni")
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * count
    running = 0.0
    for rank, (original_index, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[original_index] = running
    return tuple(adjusted)


__all__ = [
    "ConfidenceInterval",
    "PairedBootstrapResult",
    "adjust_p_values",
    "bootstrap_mean_interval",
    "paired_bootstrap",
]
