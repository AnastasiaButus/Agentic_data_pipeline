"""Tests for auto-labeling and quality checks in the annotation stage."""

from __future__ import annotations

from pathlib import Path

from src.agents.annotation_agent import AnnotationAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.providers.llm.mock_llm import MockLLM


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
    """Build a minimal pipeline context for the annotation tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(confidence_threshold=0.6),
    )
    return PipelineContext.from_config(config)


def _make_context_with_effect_labels(tmp_path: Path, effect_labels: list[str]) -> PipelineContext:
    """Build a minimal pipeline context with a custom effect-label vocabulary."""

    config = AppConfig(
        project=ProjectConfig(name="minecraft-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(confidence_threshold=0.6, effect_labels=effect_labels),
    )
    return PipelineContext.from_config(config)


def test_auto_label_adds_annotation_columns_and_preserves_canonical_schema(tmp_path: Path) -> None:
    """Auto-labeling should keep the canonical schema and add annotation columns on top."""

    agent = AnnotationAgent(_make_context(tmp_path), llm_client=MockLLM(), registry=FakeRegistry())
    df = _Frame(
        [
            {"id": "1", "source": "HF", "text": "This supplement gives me more energy.", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "2", "source": "Web", "text": "I noticed a side effect after a week.", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        ]
    )

    labeled = agent.auto_label(df)
    rows = labeled.to_dict(orient="records")

    assert list(labeled.columns) == [
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
    ]
    assert rows[0]["sentiment_label"] == "positive"
    assert rows[0]["effect_label"] == "energy"
    assert 0.0 <= rows[0]["confidence"] <= 1.0
    assert rows[1]["sentiment_label"] == "negative"
    assert rows[1]["effect_label"] == "side_effects"


def test_auto_label_uses_fallbacks_without_llm_and_handles_empty_input(tmp_path: Path) -> None:
    """The fallback path should be deterministic and empty input should keep the schema intact."""

    agent = AnnotationAgent(_make_context(tmp_path), registry=FakeRegistry())

    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "3", "source": "HF", "text": "Average result", "label": None, "rating": 3, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )
    empty = agent.auto_label(_Frame([]))

    rows = labeled.to_dict(orient="records")

    assert rows[0]["effect_label"] == "other"
    assert rows[0]["confidence"] == 0.5
    assert list(empty.columns) == [
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
    ]
    assert empty.empty


def test_check_quality_reports_core_metrics(tmp_path: Path) -> None:
    """The quality summary should expose label distribution and low-confidence counts."""

    agent = AnnotationAgent(_make_context(tmp_path), llm_client=MockLLM(), registry=FakeRegistry())
    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "This supplement gives me more energy.", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "2", "source": "Web", "text": "Nothing special here.", "label": None, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    report = agent.check_quality(labeled)

    assert report["n_rows"] == 2
    assert "label_dist" in report
    assert "confidence_mean" in report
    assert "n_low_confidence" in report


def test_auto_label_uses_configured_effect_labels(tmp_path: Path) -> None:
    """Minecraft configs should drive the effect-label vocabulary used by auto-labeling."""

    effect_labels = ["crafting", "combat", "enchantments"]
    agent = AnnotationAgent(
        _make_context_with_effect_labels(tmp_path, effect_labels),
        llm_client=MockLLM(),
        registry=FakeRegistry(),
    )

    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "1", "source": "Web", "text": "Crafting instructions for a better build.", "label": None, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "2", "source": "Web", "text": "Combat tips for the arena.", "label": None, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "3", "source": "Web", "text": "Enchantments help with progression.", "label": None, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    rows = labeled.to_dict(orient="records")

    assert [row["effect_label"] for row in rows] == effect_labels