"""Deterministic bounded agglomerative clustering of witness fingerprints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelpact.diff.witnesses import DifferenceWitness


@dataclass(frozen=True, slots=True)
class WitnessCluster:
    cluster_id: str
    witness_ids: tuple[str, ...]
    medoid_id: str
    dispersion: float
    uncertainty: float
    outlier_ids: tuple[str, ...]


def witness_vector(witness: DifferenceWitness) -> np.ndarray:
    metrics = tuple(value for _, value in sorted(witness.divergence_metrics.items()))
    values = metrics + witness.activation_fingerprint + witness.gradient_fingerprint + witness.prompt_fingerprint
    if not values:
        values = (0.0,)
    return np.asarray(values, dtype=np.float64)


def _padded_matrix(witnesses: tuple[DifferenceWitness, ...]) -> np.ndarray:
    vectors = [witness_vector(item) for item in witnesses]
    width = max(vector.size for vector in vectors)
    matrix = np.zeros((len(vectors), width), dtype=np.float64)
    for index, vector in enumerate(vectors):
        matrix[index, : vector.size] = vector
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    return (matrix - matrix.mean(axis=0)) / scales


def deterministic_agglomerative(
    witnesses: tuple[DifferenceWitness, ...],
    *,
    maximum_clusters: int = 8,
    distance_threshold: float = 2.0,
) -> tuple[WitnessCluster, ...]:
    if not witnesses:
        return ()
    ordered = tuple(sorted(witnesses, key=lambda item: item.witness_id))
    matrix = _padded_matrix(ordered)
    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(ordered))]

    def cluster_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        distances = [float(np.linalg.norm(matrix[i] - matrix[j])) for i in left for j in right]
        return float(sum(distances) / len(distances))

    while len(clusters) > 1:
        choices = [
            (cluster_distance(clusters[i], clusters[j]), clusters[i], clusters[j], i, j)
            for i in range(len(clusters))
            for j in range(i + 1, len(clusters))
        ]
        distance, left, right, left_index, right_index = min(choices, key=lambda item: (item[0], item[1], item[2]))
        if len(clusters) <= maximum_clusters and distance > distance_threshold:
            break
        merged = tuple(sorted((*left, *right)))
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {left_index, right_index}]
        clusters.append(merged)
        clusters.sort()

    results: list[WitnessCluster] = []
    for number, members in enumerate(sorted(clusters), start=1):
        pairwise = np.linalg.norm(matrix[list(members), None, :] - matrix[None, list(members), :], axis=-1)
        medoid_local = int(np.argmin(pairwise.mean(axis=1)))
        medoid = members[medoid_local]
        distances = pairwise[medoid_local]
        dispersion = float(distances.mean())
        cutoff = float(distances.mean() + 2.0 * distances.std())
        outliers = tuple(ordered[members[index]].witness_id for index, distance in enumerate(distances) if distance > cutoff and len(members) > 2)
        results.append(
            WitnessCluster(
                cluster_id=f"cluster-{number:03d}",
                witness_ids=tuple(ordered[index].witness_id for index in members),
                medoid_id=ordered[medoid].witness_id,
                dispersion=dispersion,
                uncertainty=float(distances.std()),
                outlier_ids=outliers,
            )
        )
    return tuple(results)

