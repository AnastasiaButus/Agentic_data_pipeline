"""Tests for the human review queue service."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.core.exceptions import ValidationError
from src.services.review_queue_service import ReviewQueueService


class FakeRegistry:
    """Capture review queue persistence without touching the real filesystem."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.saved: list[str] = []

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        target_path = self.root_dir / Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rows = df.to_dict(orient="records")
            columns = list(df.columns)
        except Exception:
            rows = list(df)
            columns = list(rows[0].keys()) if rows else []

        import csv
        import json

        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
            for row in rows:
                writer.writerow({column: json.dumps(row.get(column)) for column in columns})

        self.saved.append(str(path))
        return target_path

    def load_dataframe(self, path: str | Path) -> object:
        target_path = self.root_dir / Path(path)
        import csv
        import json

        with target_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [{column: self._decode_csv_value(value) for column, value in row.items()} for row in reader]

        return _Frame(rows)

    def _decode_csv_value(self, value: str) -> object:
        """Decode JSON-encoded CSV cells while tolerating plain strings in fixtures."""

        import json

        try:
            return json.loads(value)
        except Exception:
            return value

    def exists(self, path: str | Path) -> bool:
        return (self.root_dir / Path(path)).exists()


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
    """Build a minimal pipeline context for the review queue tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def _canonical_rows() -> _Frame:
    """Create a small labeled batch with a mix of confident and uncertain rows."""

    return _Frame(
        [
            {
                "id": "1",
                "source": "HF",
                "text": "Great product",
                "label": None,
                "rating": 5,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "positive",
                "effect_label": "energy",
                "confidence": 0.4,
            },
            {
                "id": "2",
                "source": "Web",
                "text": "Too many side effects",
                "label": None,
                "rating": 1,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "negative",
                "effect_label": "side_effects",
                "confidence": float("nan"),
            },
            {
                "id": "3",
                "source": "Web",
                "text": "Looks fine",
                "label": None,
                "rating": 3,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "neutral",
                "effect_label": "other",
                "confidence": 0.95,
            },
        ]
    )


def test_export_low_confidence_queue_selects_low_confidence_rows(tmp_path: Path) -> None:
    """Only rows below the threshold or missing confidence should be routed into the queue."""

    registry = FakeRegistry(tmp_path)
    service = ReviewQueueService(_make_context(tmp_path), registry=registry)

    queue = service.export_low_confidence_queue(_canonical_rows(), threshold=0.7)
    rows = queue.to_dict(orient="records")

    assert [row["id"] for row in rows] == ["1", "2"]
    assert registry.saved == ["data/interim/review_queue.csv"]


def test_export_low_confidence_queue_adds_review_columns_and_defaults(tmp_path: Path) -> None:
    """The queue export should add the review columns with deterministic defaults."""

    service = ReviewQueueService(_make_context(tmp_path), registry=FakeRegistry(tmp_path))

    queue = service.export_low_confidence_queue(_canonical_rows(), threshold=0.7)
    row = queue.to_dict(orient="records")[0]

    assert row["reviewed_effect_label"] == ""
    assert row["review_comment"] == ""
    assert row["human_verified"] is False
    assert "effect_label" in row and "confidence" in row


def test_merge_reviewed_labels_replaces_effect_label_only_for_real_corrections(tmp_path: Path) -> None:
    """Only non-empty reviewed effect labels should replace the original effect label."""

    service = ReviewQueueService(_make_context(tmp_path), registry=FakeRegistry(tmp_path))
    original = _canonical_rows()
    corrected = _Frame(
        [
            {
                "id": "1",
                "reviewed_effect_label": "other",
                "review_comment": "Looks fine",
                "human_verified": False,
            },
            {
                "id": "2",
                "reviewed_effect_label": "",
                "review_comment": "",
                "human_verified": False,
            },
        ]
    )

    merged = service.merge_reviewed_labels(original, corrected)
    rows = merged.to_dict(orient="records")
    row_one = next(row for row in rows if row["id"] == "1")
    row_two = next(row for row in rows if row["id"] == "2")

    assert row_one["effect_label"] == "other"
    assert row_one["confidence"] == 1.0
    assert row_one["human_verified"] is True
    assert row_two["effect_label"] == "side_effects"
    assert row_two["human_verified"] is False


def test_merge_reviewed_labels_validates_duplicate_and_unknown_ids(tmp_path: Path) -> None:
    """Duplicate or unknown ids in the corrected queue should fail fast."""

    service = ReviewQueueService(_make_context(tmp_path), registry=FakeRegistry(tmp_path))
    original = _canonical_rows()

    duplicate_corrected = _Frame(
        [
            {"id": "1", "reviewed_effect_label": "other"},
            {"id": "1", "reviewed_effect_label": "energy"},
        ]
    )
    unknown_corrected = _Frame(
        [
            {"id": "999", "reviewed_effect_label": "other"},
        ]
    )

    with pytest.raises(ValidationError):
        service.merge_reviewed_labels(original, duplicate_corrected)

    with pytest.raises(ValidationError):
        service.merge_reviewed_labels(original, unknown_corrected)


def test_merge_reviewed_labels_rejects_corrected_rows_when_original_is_empty(tmp_path: Path) -> None:
    """Corrected rows without any original data should still fail validation."""

    service = ReviewQueueService(_make_context(tmp_path), registry=FakeRegistry(tmp_path))
    empty_original = _Frame([])
    corrected = _Frame(
        [
            {"id": "1", "reviewed_effect_label": "other"},
        ]
    )

    with pytest.raises(ValidationError):
        service.merge_reviewed_labels(empty_original, corrected)


def test_merge_reviewed_labels_returns_empty_frame_when_both_inputs_are_empty(tmp_path: Path) -> None:
    """Empty original and corrected inputs should return an empty merged frame with the merged schema."""

    service = ReviewQueueService(_make_context(tmp_path), registry=FakeRegistry(tmp_path))

    merged = service.merge_reviewed_labels(_Frame([]), _Frame([]))

    assert merged.empty
    assert list(merged.columns) == [
        "id",
        "source",
        "text",
        "label",
        "rating",
        "created_at",
        "split",
        "meta_json",
        "sentiment_label",
        "effect_label",
        "confidence",
        "reviewed_effect_label",
        "review_comment",
        "human_verified",
    ]


def test_load_corrected_queue_reads_default_path_and_fails_when_missing(tmp_path: Path) -> None:
    """The default corrected queue path should be readable and missing files should fail clearly."""

    registry = FakeRegistry(tmp_path)
    service = ReviewQueueService(_make_context(tmp_path), registry=registry)

    corrected_path = tmp_path / "data" / "interim" / "review_queue_corrected.csv"
    corrected_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_path.write_text((Path(__file__).resolve().parents[1] / "fixtures" / "sample_review_queue_corrected.csv").read_text(encoding="utf-8"), encoding="utf-8")

    loaded = service.load_corrected_queue()

    assert [str(row["id"]) for row in loaded.to_dict(orient="records")] == ["1", "2"]

    corrected_path.unlink()
    with pytest.raises(FileNotFoundError):
        service.load_corrected_queue()