"""Tests for quality fixing and comparison."""

from __future__ import annotations

from pathlib import Path

from src.agents.data_quality_agent import DataQualityAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.core.constants import STANDARD_COLUMNS
from src.utils.text_cleaning import normalize_whitespace, safe_word_count


class FakeRegistry:
    """Capture artifact writes without touching the filesystem."""

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        return Path(path)


class _Frame:
    """Tiny dataframe-like helper used to keep the test independent of pandas."""

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
    """Build a minimal pipeline context for the quality tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def test_text_cleaning_helpers_handle_none_blank_and_spaced_text() -> None:
    """The text cleaning helpers should stay safe on None and whitespace-only input."""

    assert normalize_whitespace(None) == ""
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   ") == ""
    assert normalize_whitespace("  Good   product  review  ") == "Good product review"
    assert safe_word_count(None) == 0
    assert safe_word_count("") == 0
    assert safe_word_count("   ") == 0
    assert safe_word_count("  Good   product  review  ") == 3


def test_fix_removes_empty_short_and_duplicate_rows_and_preserves_schema(tmp_path: Path) -> None:
    """The fix step should apply the requested quality strategy without breaking the canonical schema."""

    agent = DataQualityAgent(_make_context(tmp_path), registry=FakeRegistry())
    dirty = _Frame(
        [
            {"id": "1", "source": "HF", "text": "  Protein   powder  review  ", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}", "extra": "x"},
            {"id": "2", "source": "HF", "text": "Protein powder review", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "3", "source": "HF", "text": "", "label": "positive", "rating": 4, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "4", "source": "Web", "text": "ok", "label": "neutral", "rating": 3, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "5", "source": "Web", "text": "Nice concise review", "label": "neutral", "rating": 3, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "6", "source": "Web", "text": "Another helpful review", "label": "neutral", "rating": 4, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "7", "source": "Web", "text": " ".join(["long"] * 70), "label": "negative", "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        ]
    )

    cleaned = agent.fix(
        dirty,
        {
            "drop_empty_text": True,
            "min_words": 3,
            "normalize_whitespace": True,
            "duplicates": "drop",
            "outliers": "remove_iqr",
        },
    )

    rows = cleaned.to_dict(orient="records")

    assert list(cleaned.columns) == list(STANDARD_COLUMNS)
    assert len(rows) == 3
    assert [row["id"] for row in rows] == ["1", "5", "6"]
    assert rows[0]["text"] == "Protein powder review"
    assert all(column in rows[0] for column in STANDARD_COLUMNS)


def test_fix_can_clip_iqr_outlier_texts(tmp_path: Path) -> None:
    """The clip_iqr strategy should shorten long text while keeping the row."""

    agent = DataQualityAgent(_make_context(tmp_path), registry=FakeRegistry())
    dirty = _Frame(
        [
            {"id": "1", "source": "HF", "text": "Short helpful review", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "2", "source": "HF", "text": "Useful fitness supplement review", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "3", "source": "Web", "text": "Clear and concise product feedback", "label": "positive", "rating": 4, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "4", "source": "Web", "text": "Direct and practical supplement note", "label": "negative", "rating": 2, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "5", "source": "Web", "text": " ".join(["long"] * 80), "label": "negative", "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        ]
    )

    cleaned = agent.fix(
        dirty,
        {
            "drop_empty_text": True,
            "min_words": 1,
            "normalize_whitespace": True,
            "duplicates": "drop",
            "outliers": "clip_iqr",
        },
    )

    rows = cleaned.to_dict(orient="records")
    clipped_row = next(row for row in rows if row["id"] == "5")

    assert list(cleaned.columns) == list(STANDARD_COLUMNS)
    assert safe_word_count(clipped_row["text"]) < 80
    assert clipped_row["text"]


def test_compare_reports_expected_metrics(tmp_path: Path) -> None:
    """The compare table should expose the expected metric names and values."""

    agent = DataQualityAgent(_make_context(tmp_path), registry=FakeRegistry())
    before = _Frame(
        [
            {"id": "1", "source": "HF", "text": "Great product", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "2", "source": "HF", "text": "Great product", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "3", "source": "Web", "text": "   ", "label": "negative", "rating": 2, "created_at": "now", "split": None, "meta_json": "{}"},
        ]
    )
    after = agent.fix(
        before,
        {
            "drop_empty_text": True,
            "min_words": 1,
            "normalize_whitespace": True,
            "duplicates": "drop",
            "outliers": "remove_iqr",
        },
    )

    comparison = agent.compare(before, after)
    rows = comparison.to_dict(orient="records")
    metrics = {row["metric"]: row for row in rows}

    assert [row["metric"] for row in rows] == ["n_rows", "duplicates", "empty_text", "mean_words"]
    assert metrics["n_rows"]["before"] == 3.0
    assert metrics["n_rows"]["after"] == 1.0
    assert metrics["duplicates"]["before"] == 1.0
    assert metrics["duplicates"]["after"] == 0.0
    assert metrics["empty_text"]["before"] == 1.0
    assert metrics["empty_text"]["after"] == 0.0