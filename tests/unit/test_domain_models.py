"""Tests for the domain models and integration contracts."""

from __future__ import annotations

import json

import pytest

from src.domain import (
    ALIterationResult,
    AnnotationSpec,
    ComparisonMetric,
    LabelResult,
    QualityReport,
    SourceCandidate,
    TrainingMetrics,
)


def test_domain_dataclasses_create_with_valid_values() -> None:
    """The domain dataclasses should accept valid values for the Step 2 schema."""

    candidate = SourceCandidate(
        source_id="hf_reviews",
        source_type="hf_dataset",
        title="Fitness supplements reviews",
        uri="fitness-supplements/reviews",
        score=0.8,
        metadata={"split": "train"},
    )
    quality = QualityReport(
        missing={"rating": 0.1},
        duplicates=2,
        outliers={"length": {"count": 3}},
        imbalance={"label": {"positive": 0.7, "negative": 0.3}},
        warnings=["low coverage"],
    )
    spec = AnnotationSpec(
        name="fitness_review_spec",
        description="Annotation guide for supplements reviews",
        text_field="text",
        label_field="label",
        id_field="id",
        labels=["positive", "negative"],
        instructions=["Read the review carefully."],
        output_format="markdown",
    )
    label_result = LabelResult(label="positive", confidence=0.5, rationale="Clear benefit signal.")
    al_result = ALIterationResult(iteration=1, n_labeled=64, accuracy=0.82, f1=0.79)
    comparison = ComparisonMetric(name="f1", current=0.79, baseline=0.75, delta=0.04)
    training = TrainingMetrics(
        accuracy=0.91,
        precision=0.9,
        recall=0.89,
        f1=0.895,
        loss=0.12,
        n_examples=128,
        extra={"auc": 0.95},
    )

    assert candidate.source_type == "hf_dataset"
    assert quality.duplicates == 2
    assert spec.labels == ["positive", "negative"]
    assert label_result.label == "positive"
    assert al_result.iteration == 1
    assert comparison.delta == 0.04
    assert training.n_examples == 128


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_label_result_confidence_bounds(confidence: float) -> None:
    """LabelResult should accept both inclusive confidence boundaries."""

    result = LabelResult(label="neutral", confidence=confidence)

    assert result.confidence == confidence


def test_quality_report_and_iteration_dicts_cover_expected_keys() -> None:
    """The report dataclasses should serialize to dictionaries with the requested keys."""

    quality = QualityReport()
    iteration = ALIterationResult(iteration=2, n_labeled=20, accuracy=0.7, f1=0.65)

    quality_dict = quality.as_dict()
    iteration_dict = iteration.as_dict()

    assert set(quality_dict) == {"missing", "duplicates", "outliers", "imbalance", "warnings"}
    assert iteration_dict["iteration"] == 2
    assert iteration_dict["n_labeled"] == 20
    assert iteration_dict["accuracy"] == 0.7
    assert iteration_dict["f1"] == 0.65


def test_source_candidate_preserves_api_metadata_and_empty_containers() -> None:
    """SourceCandidate should preserve source type, metadata, and empty defaults."""

    api_candidate = SourceCandidate(
        source_id="internal_reviews_api",
        source_type="api",
        title="Hidden API reviews endpoint",
        uri="https://example.com/api/reviews",
        metadata={},
    )

    assert api_candidate.source_type == "api"
    assert api_candidate.metadata == {}
    assert api_candidate.score == 0.0


def test_serialization_helpers_return_json_friendly_dicts() -> None:
    """The helper dictionaries should remain JSON-serializable and stable for empty values."""

    quality = QualityReport(warnings=[], imbalance={})
    candidate = SourceCandidate(
        source_id="scrape_1",
        source_type="scrape",
        title="Scraped reviews",
        uri="https://example.com/reviews",
        metadata={},
    )
    training = TrainingMetrics()

    assert quality.warnings == []
    assert quality.imbalance == {}
    assert candidate.metadata == {}
    assert json.loads(json.dumps(training.as_dict())) == training.as_dict()
