"""Unit tests for auto-vs-human agreement reporting on the reviewed subset."""

from __future__ import annotations

import pytest

from src.services.review_agreement import build_review_agreement_summary


def test_review_agreement_is_unavailable_without_corrected_queue() -> None:
    """Missing corrected queue should yield a truthful unavailable agreement summary."""

    summary = build_review_agreement_summary(
        annotated=[{"id": "1", "effect_label": "energy", "text": "boost", "confidence": 0.4}],
        corrected_queue=None,
    )

    assert summary["corrected_queue_found"] is False
    assert summary["compared_rows"] == 0
    assert summary["agreement"] is None
    assert summary["kappa"] is None
    assert summary["kappa_status"] == "not_available_no_compared_rows"
    assert "auto-vs-human agreement" in summary["notes"][0]


def test_review_agreement_uses_original_auto_labels_when_corrected_queue_has_only_human_labels() -> None:
    """Agreement should join corrected rows back to the original annotated data by id."""

    annotated = [
        {"id": "1", "source": "HF", "text": "boost", "effect_label": "energy", "confidence": 0.4},
        {"id": "2", "source": "Web", "text": "rash", "effect_label": "side_effects", "confidence": 0.3},
        {"id": "3", "source": "HF", "text": "neutral", "effect_label": "other", "confidence": 0.8},
    ]
    corrected_queue = [
        {"id": "1", "reviewed_effect_label": "energy"},
        {"id": "2", "reviewed_effect_label": "energy"},
        {"id": "3", "reviewed_effect_label": "other"},
    ]

    summary = build_review_agreement_summary(annotated, corrected_queue)

    assert summary["corrected_queue_found"] is True
    assert summary["n_corrected_rows"] == 3
    assert summary["n_reviewed_rows"] == 3
    assert summary["compared_rows"] == 3
    assert summary["matched_rows"] == 2
    assert summary["disagreement_rows"] == 1
    assert summary["agreement"] == pytest.approx(2 / 3)
    assert summary["kappa"] is None or -1.0 <= summary["kappa"] <= 1.0
    assert summary["auto_label_distribution"] == {"energy": 1, "side_effects": 1, "other": 1}
    assert summary["human_label_distribution"] == {"energy": 2, "other": 1}
    assert summary["disagreement_examples"][0]["id"] == "2"
    assert summary["disagreement_examples"][0]["auto_effect_label"] == "side_effects"
    assert summary["disagreement_examples"][0]["reviewed_effect_label"] == "energy"


def test_review_agreement_skips_blank_ids_and_empty_review_labels() -> None:
    """Only valid ids with reviewed_effect_label should contribute to agreement metrics."""

    annotated = [{"id": "valid-1", "effect_label": "energy", "text": "boost", "confidence": 0.4}]
    corrected_queue = [
        {"id": "", "reviewed_effect_label": "other"},
        {"id": None, "reviewed_effect_label": "energy"},
        {"id": "valid-1", "reviewed_effect_label": ""},
    ]

    summary = build_review_agreement_summary(annotated, corrected_queue)

    assert summary["n_corrected_rows"] == 1
    assert summary["n_reviewed_rows"] == 0
    assert summary["compared_rows"] == 0
    assert summary["agreement"] is None
    assert summary["disagreement_examples"] == []
