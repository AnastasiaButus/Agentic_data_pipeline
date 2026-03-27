"""Tests for annotation spec generation and rating mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.annotation_agent import AnnotationAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
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
    """Build a minimal pipeline context for the annotation tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
        request=RequestConfig(topic="fitness supplements", domain="supplements", modality="text"),
    )
    return PipelineContext.from_config(config)


def _make_context_with_effect_labels(tmp_path: Path, effect_labels: list[str]) -> PipelineContext:
    """Build a minimal pipeline context with a custom effect-label vocabulary."""

    config = AppConfig(
        project=ProjectConfig(name="minecraft-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(effect_labels=effect_labels),
        request=RequestConfig(topic="minecraft instructions", domain="minecraft", modality="text"),
    )
    return PipelineContext.from_config(config)


@pytest.mark.parametrize(
    "rating, expected",
    [(1, "negative"), (3, "neutral"), (5, "positive")],
)
def test_rating_to_sentiment_mapping(rating: int, expected: str) -> None:
    """The rating-to-sentiment mapping should stay deterministic for the main scale anchors."""

    agent = AnnotationAgent(_make_context(Path(".")), registry=FakeRegistry())

    assert agent.map_rating_to_sentiment(rating) == expected


@pytest.mark.parametrize("rating", [None, float("nan")])
def test_rating_to_sentiment_handles_missing_values(rating: object) -> None:
    """Missing or NaN ratings should not crash the mapping logic."""

    agent = AnnotationAgent(_make_context(Path(".")), registry=FakeRegistry())

    assert agent.map_rating_to_sentiment(rating) is None


def test_generate_spec_contains_required_sections(tmp_path: Path) -> None:
    """The generated spec should be human-readable and aligned to the canonical text/id/label fields."""

    agent = AnnotationAgent(_make_context(tmp_path), registry=FakeRegistry())

    spec = agent.generate_spec(
        _Frame([
            {"id": "1", "text": "Great product", "label": None, "rating": 5, "source": "HF", "created_at": "now", "split": None, "meta_json": "{}"},
        ]),
        task="fitness supplements annotation",
    )

    assert "Task" in spec
    assert "Classes and definitions" in spec
    assert "Examples" in spec
    assert "Edge cases" in spec
    assert "text" in spec
    assert "label" in spec
    assert "id" in spec


def test_generate_spec_uses_configured_effect_labels(tmp_path: Path) -> None:
    """The annotation spec should reflect the configured effect-label vocabulary."""

    effect_labels = ["crafting", "combat", "enchantments"]
    agent = AnnotationAgent(_make_context_with_effect_labels(tmp_path, effect_labels), registry=FakeRegistry())

    spec = agent.generate_spec(
        _Frame([{"id": "1", "text": "Crafting instructions", "label": None, "rating": None, "source": "Web", "created_at": "now", "split": None, "meta_json": "{}"}]),
        task="minecraft instructions annotation",
    )

    for label in effect_labels:
        assert label in spec
    assert "configured default label" in spec
