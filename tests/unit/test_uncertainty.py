"""Tests for active learning uncertainty helpers."""

from __future__ import annotations

from src.ml.uncertainty import entropy_sampling, margin_sampling, random_sampling


def test_entropy_sampling_ranks_highest_uncertainty_above_certain_rows() -> None:
    """Entropy should score uniform probabilities higher than peaked ones."""

    proba = [
        [0.9, 0.1],
        [0.5, 0.5],
    ]

    scores = entropy_sampling(proba)

    assert scores[1] > scores[0]


def test_margin_sampling_ranks_smallest_margin_as_most_uncertain() -> None:
    """Margin sampling should score near-tied probabilities higher than separated ones."""

    proba = [
        [0.85, 0.15],
        [0.51, 0.49],
    ]

    scores = margin_sampling(proba)

    assert scores[1] > scores[0]


def test_random_sampling_returns_unique_indices_with_batch_cap() -> None:
    """Random sampling should never return duplicate indices and should respect the batch cap."""

    indices = random_sampling(n=10, batch_size=5, random_state=42)

    assert len(indices) == 5
    assert len(set(indices)) == 5
    assert all(0 <= index < 10 for index in indices)