"""Unit tests for the richer EDA reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
from src.core.context import PipelineContext
from src.services.reporting_service import ReportingService


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
    """Build a minimal pipeline context for reporting tests."""

    config = AppConfig(
        project=ProjectConfig(name="reporting-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(effect_labels=["energy", "side_effects", "other"]),
        request=RequestConfig(topic="fitness supplements", domain="supplements", modality="text"),
    )
    return PipelineContext.from_config(config)


def test_eda_context_includes_extended_metrics(tmp_path: Path) -> None:
    """The machine-readable EDA context should expose richer summary blocks."""

    service = ReportingService(_make_context(tmp_path))
    raw = _Frame(
        [
            {"id": "1", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "2", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "3", "source": "Web", "text": "side effect warning", "rating": 1},
        ]
    )
    cleaned = _Frame(
        [
            {"id": "1", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "3", "source": "Web", "text": "side effect warning", "rating": 1},
        ]
    )

    context_path = service.write_eda_context(
        cleaned,
        raw_df_like=raw,
        quality_report={"warnings": ["duplicates removed"]},
    )
    payload = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert payload["n_rows"] == 2
    assert payload["column_count"] == 4
    assert payload["raw_vs_cleaned"]["available"] is True
    assert payload["raw_vs_cleaned"]["raw_rows"] == 3
    assert payload["raw_vs_cleaned"]["cleaned_rows"] == 2
    assert payload["duplicate_summary"]["available"] is True
    assert payload["duplicate_summary"]["duplicate_rows"] == 0
    assert payload["rating_distribution"]["available"] is True
    assert payload["text_length_buckets"]["available"] is True
    assert payload["quality_warnings"] == ["duplicates removed"]


def test_eda_html_report_is_created(tmp_path: Path) -> None:
    """The HTML EDA export should create a self-contained report file."""

    service = ReportingService(_make_context(tmp_path))
    cleaned = _Frame(
        [
            {"id": "1", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "2", "source": "Web", "text": "side effect warning", "rating": 1},
        ]
    )

    html_path = service.write_eda_html_report(cleaned, raw_df_like=cleaned, quality_report={"warnings": []})
    html = (tmp_path / html_path).read_text(encoding="utf-8")

    assert "<html" in html.lower()
    assert "EDA Report" in html
    assert "Raw vs cleaned" in html
    assert "Charts" in html


def test_review_queue_report_and_context_make_hitl_steps_explicit(tmp_path: Path) -> None:
    """The review queue artifacts should tell the reviewer what to do next."""

    service = ReportingService(_make_context(tmp_path))
    review_queue = _Frame(
        [
            {
                "id": "1",
                "source": "HF",
                "text": "energy boost",
                "label": None,
                "effect_label": "energy",
                "confidence": 0.42,
            }
        ]
    )

    report_path = service.write_review_queue_report(review_queue, 0.6, ["energy", "side_effects", "other"])
    context_path = service.write_review_queue_context(review_queue, 0.6, ["energy", "side_effects", "other"])

    report = (tmp_path / report_path).read_text(encoding="utf-8")
    payload = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert "## Reviewer guide" in report
    assert "## To-do reviewer" in report
    assert "review_queue_corrected.csv" in report
    assert "## Next step" in report
    assert payload["review_required"] is True
    assert payload["current_stage"] == "human_review"
    assert payload["next_step"] == "fill_corrected_queue_and_rerun"
    assert "reviewed_effect_label" in payload["review_columns"]


def test_review_merge_report_explains_next_step(tmp_path: Path) -> None:
    """The merge report should explain what happens after HITL merge."""

    service = ReportingService(_make_context(tmp_path))

    report_path = service.write_review_merge_report(
        corrected_queue_found=True,
        corrected_queue_path="data/interim/review_queue_corrected.csv",
        n_corrected_rows=1,
        n_rows_with_reviewed_effect_label=1,
        n_effect_label_changes=1,
        reviewed_effect_labels=["side_effects"],
        review_status="merged",
    )
    report = (tmp_path / report_path).read_text(encoding="utf-8")

    assert "## Next step" in report
    assert "retrain / active learning" in report
