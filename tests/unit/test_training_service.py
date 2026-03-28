"""Tests for the training service baseline and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.core.exceptions import ValidationError
from src.services.training_service import TrainingService


class _Frame:
    """Tiny dataframe-like helper used to keep the tests independent of pandas."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    @property
    def empty(self) -> bool:
        return not self._rows

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        if orient != "records":
            raise ValueError("Only records orientation is supported")
        return [dict(row) for row in self._rows]


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for training tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def _training_rows() -> _Frame:
    """Create a small multiclass dataset suitable for the training baseline."""

    rows = []
    labels = ["energy", "side_effects", "other"]
    for index in range(30):
        label = labels[index % len(labels)]
        rows.append(
            {
                "id": str(index + 1),
                "source": "HF",
                "text": f"Training review {index} about {label} and supplements",
                "label": None,
                "rating": 5 if label == "energy" else (1 if label == "side_effects" else 3),
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": None,
                "effect_label": label,
                "confidence": 1.0,
            }
        )
    return _Frame(rows)


def test_training_service_train_returns_accuracy_and_f1_and_saves_metrics(tmp_path: Path) -> None:
    """Training should return the requested metrics and persist model_metrics.json."""

    service = TrainingService(_make_context(tmp_path))

    artifacts, metrics = service.train(_training_rows())

    assert {"accuracy", "f1"}.issubset(metrics.keys())
    assert Path(artifacts["metrics_path"]).exists()
    assert Path(artifacts["model_path"]).exists()
    assert Path(artifacts["vectorizer_path"]).exists()


def test_training_service_raises_on_empty_or_missing_target(tmp_path: Path) -> None:
    """Training should fail clearly when effect_label is missing after filtering."""

    service = TrainingService(_make_context(tmp_path))

    with pytest.raises(ValidationError):
        service.train(_Frame([]))

    with pytest.raises(ValidationError):
        service.train(
            _Frame(
                [
                    {"id": "1", "source": "HF", "text": "Training review", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}", "sentiment_label": None, "effect_label": "", "confidence": 1.0},
                ]
            )
        )


def test_training_service_uses_fallback_split_on_small_dataset(tmp_path: Path) -> None:
    """Very small datasets should still split deterministically instead of failing silently."""

    service = TrainingService(_make_context(tmp_path))
    small_rows = _Frame(
        [
            {"id": "1", "source": "HF", "text": "Energy boost", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}", "sentiment_label": None, "effect_label": "energy", "confidence": 1.0},
            {"id": "2", "source": "HF", "text": "Side effect note", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}", "sentiment_label": None, "effect_label": "side_effects", "confidence": 1.0},
            {"id": "3", "source": "HF", "text": "Neutral review", "label": None, "rating": 3, "created_at": "now", "split": None, "meta_json": "{}", "sentiment_label": None, "effect_label": "other", "confidence": 1.0},
            {"id": "4", "source": "HF", "text": "More energy", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}", "sentiment_label": None, "effect_label": "energy", "confidence": 1.0},
            {"id": "5", "source": "HF", "text": "Another effect", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}", "sentiment_label": None, "effect_label": "side_effects", "confidence": 1.0},
        ]
    )

    artifacts, metrics = service.train(small_rows)

    assert {"accuracy", "f1"}.issubset(metrics.keys())
    assert Path(artifacts["metrics_path"]).exists()


def test_training_service_compares_baseline_and_reviewed_retrain(tmp_path: Path) -> None:
    """Comparison helper should report baseline/reviewed metrics and metric deltas."""

    service = TrainingService(_make_context(tmp_path))
    baseline = _training_rows()
    reviewed = _training_rows()

    summary = service.compare_baseline_and_reviewed(baseline, reviewed)

    assert summary["comparison_scope"] == "auto_labeled_baseline_vs_reviewed_retrain"
    assert summary["baseline_status"] == "computed"
    assert summary["reviewed_status"] == "computed"
    assert summary["datasets_identical"] is True
    assert summary["baseline_metrics"]["n_examples"] == 30
    assert summary["reviewed_metrics"]["n_examples"] == 30
    assert summary["delta_accuracy"] == 0.0
    assert summary["delta_f1"] == 0.0


def test_training_service_comparison_reports_validation_failure_for_invalid_reviewed_dataset(tmp_path: Path) -> None:
    """Comparison helper should keep baseline metrics and explain why reviewed retrain is unavailable."""

    service = TrainingService(_make_context(tmp_path))
    baseline = _training_rows()
    reviewed = _Frame(
        [
            {
                "id": "1",
                "source": "HF",
                "text": "Only one class remains here",
                "label": None,
                "rating": 5,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": None,
                "effect_label": "energy",
                "confidence": 1.0,
            }
        ]
    )

    summary = service.compare_baseline_and_reviewed(baseline, reviewed)

    assert summary["baseline_status"] == "computed"
    assert summary["reviewed_status"] == "not_available_validation_error"
    assert summary["delta_accuracy"] is None
    assert summary["delta_f1"] is None
    assert "at least two effect_label classes" in summary["reviewed_error"]
