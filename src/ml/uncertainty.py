"""Uncertainty sampling utilities for active learning."""

from __future__ import annotations

import math
import random
from typing import Iterable


def entropy_sampling(proba: Iterable[Iterable[float]]) -> list[float]:
    """Return entropy-based uncertainty scores for each row of class probabilities."""

    rows = _as_rows(proba)
    scores: list[float] = []
    for row in rows:
        if not row:
            scores.append(0.0)
            continue

        safe_row = [max(float(value), 0.0) for value in row]
        total = sum(safe_row)
        if total <= 0:
            scores.append(0.0)
            continue

        normalized = [value / total for value in safe_row]
        entropy = 0.0
        for probability in normalized:
            if probability > 0:
                entropy -= probability * math.log(probability)
        scores.append(entropy)
    return scores


def margin_sampling(proba: Iterable[Iterable[float]]) -> list[float]:
    """Return margin-based uncertainty scores where larger means more uncertain."""

    rows = _as_rows(proba)
    scores: list[float] = []
    for row in rows:
        if len(row) < 2:
            scores.append(0.0)
            continue
        sorted_row = sorted((max(float(value), 0.0) for value in row), reverse=True)
        margin = max(sorted_row[0] - sorted_row[1], 0.0)
        scores.append(max(1.0 - margin, 0.0))
    return scores


def random_sampling(n: int, batch_size: int, random_state: int | None = None) -> list[int]:
    """Return a unique random subset of indices, capped at the available population size."""

    if n <= 0 or batch_size <= 0:
        return []

    rng = random.Random(random_state)
    population = list(range(n))
    return rng.sample(population, k=min(n, batch_size))


def _as_rows(proba: Iterable[Iterable[float]]) -> list[list[float]]:
    """Materialize probability inputs into a list of row lists."""

    return [[float(value) for value in row] for row in proba]