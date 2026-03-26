"""Tests for schema normalization of raw review rows."""

from __future__ import annotations

import json

import pytest

from src.core.constants import STANDARD_COLUMNS
from src.core.exceptions import ValidationError
from src.services import schema_normalization_service as schema_module
from src.services.schema_normalization_service import SchemaNormalizationService


def test_normalize_reviews_creates_all_standard_columns() -> None:
    """Normalized rows should always include the shared review schema columns."""

    service = SchemaNormalizationService()
    frame = service.normalize_reviews(
        [
            {"text": "Great product", "rating": 5, "raw_label": "positive", "extra": "x"},
        ],
        source_name="reviews_a",
        source_type="api",
    )

    assert list(frame.columns) == list(STANDARD_COLUMNS)
    assert len(frame.columns) == len(set(frame.columns))
    assert "meta_json" in STANDARD_COLUMNS


def test_normalize_reviews_uses_text_fallback_columns() -> None:
    """The loader should fall back to content and review_text when text is absent."""

    service = SchemaNormalizationService()
    frame = service.normalize_reviews(
        [{"review_text": "Fallback text", "rating": 4}],
        source_name="reviews_b",
        source_type="hf_dataset",
    )

    rows = frame.to_dict(orient="records")
    assert rows[0]["text"] == "Fallback text"
    assert list(frame.columns) == list(STANDARD_COLUMNS)


def test_normalize_reviews_output_schema_has_no_duplicate_columns() -> None:
    """The normalized output should be strictly aligned to the standard schema."""

    service = SchemaNormalizationService()
    frame = service.normalize_reviews(
        [
            {
                "content": "Useful review",
                "rating": 3,
                "vendor": "acme",
                "source_url": "https://example.com/item",
            }
        ],
        source_name="reviews_c",
        source_type="scrape",
    )

    assert list(frame.columns) == list(STANDARD_COLUMNS)
    assert len(frame.columns) == len(set(frame.columns))
    assert all(column in STANDARD_COLUMNS for column in frame.columns)


def test_normalize_reviews_maps_values_into_standard_columns() -> None:
    """Normalized values should land in the correct standard fields, not alias columns."""

    service = SchemaNormalizationService()
    frame = service.normalize_reviews(
        [
            {
                "text": "Great product",
                "rating": 5,
                "raw_label": "positive",
                "language": "en",
                "product_name": "Protein Powder",
            }
        ],
        source_name="reviews_values",
        source_type="api",
    )

    row = frame.to_dict(orient="records")[0]

    assert row["source"] == "reviews_values"
    assert row["text"] == "Great product"
    assert row["label"] == "positive"
    assert row["rating"] == 5
    assert row["split"] is None
    assert isinstance(row["created_at"], str)
    assert isinstance(row["meta_json"], str)


def test_normalize_reviews_preserves_source_name_and_source_type_in_meta_json() -> None:
    """Source metadata should be preserved even though only the standard columns are emitted."""

    service = SchemaNormalizationService()
    frame = service.normalize_reviews(
        [{"content": "Useful review", "rating": 3}],
        source_name="reviews_meta",
        source_type="scrape",
    )

    row = frame.to_dict(orient="records")[0]
    meta = json.loads(row["meta_json"])

    assert meta["source_name"] == "reviews_meta"
    assert meta["source_type"] == "scrape"
    assert meta["content"] == "Useful review"
    assert meta["record_id"] == row["id"]
    assert meta["confidence"] is None

def test_normalize_reviews_meta_json_available_in_canonical_schema() -> None:
    """meta_json should be part of the canonical normalized schema."""

    service = SchemaNormalizationService()
    frame = service.normalize_reviews(
        [{"text": "Great product", "rating": 5, "vendor": "acme"}],
        source_name="reviews_schema",
        source_type="api",
    )

    assert "meta_json" in frame.columns
    row = frame.to_dict(orient="records")[0]
    assert json.loads(row["meta_json"])["vendor"] == "acme"


def test_normalize_reviews_empty_frame_returns_standard_columns() -> None:
    """An empty input frame should return an empty normalized frame with canonical columns."""

    service = SchemaNormalizationService()
    frame = service.normalize_reviews([], source_name="reviews_d", source_type="api")

    assert frame.empty
    assert list(frame.columns) == list(STANDARD_COLUMNS)


def test_normalize_reviews_without_content_columns_raises_validation_error() -> None:
    """Missing content columns should fail fast with a ValidationError."""

    service = SchemaNormalizationService()

    with pytest.raises(ValidationError):
        service.normalize_reviews([{"rating": 5}], source_name="reviews_e", source_type="api")


def test_normalize_reviews_uses_distinct_ids_for_same_text_across_sources() -> None:
    """The stable record id should vary across different source names."""

    service = SchemaNormalizationService()
    left = service.normalize_reviews([{"text": "Same text"}], source_name="source_a", source_type="api")
    right = service.normalize_reviews([{"text": "Same text"}], source_name="source_b", source_type="api")

    left_id = left.to_dict(orient="records")[0]["id"]
    right_id = right.to_dict(orient="records")[0]["id"]

    assert left_id != right_id

def test_simple_dataframe_respects_explicit_columns_in_to_dict() -> None:
    """The fallback dataframe shim should filter records to the provided columns like pandas."""

    frame = schema_module.pd.DataFrame(
        [
            {"keep": 1, "drop": 2},
            {"keep": 3, "drop": 4},
        ],
        columns=["keep"],
    )

    assert frame.to_dict(orient="records") == [{"keep": 1}, {"keep": 3}]
