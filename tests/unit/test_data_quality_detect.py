"""Tests for quality issue detection."""

from __future__ import annotations

from pathlib import Path

from src.agents.data_quality_agent import DataQualityAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext


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


def test_detect_issues_finds_missing_duplicates_outliers_and_imbalance(tmp_path: Path) -> None:
    """Dirty canonical rows should surface missing values, duplicate text, outliers, and imbalance."""

    agent = DataQualityAgent(_make_context(tmp_path), registry=FakeRegistry())
    dirty = _Frame(
        [
            {"id": "1", "source": "HF", "text": None, "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "2", "source": "HF", "text": "", "label": "positive", "rating": 4, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "3", "source": "HF", "text": "   ", "label": "positive", "rating": 4, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "4", "source": "HF", "text": "Great product", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "5", "source": "Web", "text": "Great product", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "6", "source": "Web", "text": "Helpful and concise review", "label": "positive", "rating": 4, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "7", "source": "Web", "text": " ".join(["long"] * 60), "label": "negative", "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        ]
    )

    report = agent.detect_issues(dirty)

    assert report.missing["text"] == 3 / 7
    assert report.duplicates == 1
    assert report.outliers["text"]["count"] == 1
    assert report.imbalance["label"]["distribution"]["positive"] == 6 / 7
    assert report.warnings
    assert any("missing" in warning for warning in report.warnings)
    assert any("duplicate" in warning for warning in report.warnings)
    assert any("outlier" in warning for warning in report.warnings)
    assert any("imbalance" in warning for warning in report.warnings)


def test_detect_issues_handles_empty_input(tmp_path: Path) -> None:
    """An empty frame should produce a safe report rather than raising."""

    agent = DataQualityAgent(_make_context(tmp_path), registry=FakeRegistry())

    report = agent.detect_issues(_Frame([]))

    assert report.duplicates == 0
    assert report.warnings == ["empty dataset"]