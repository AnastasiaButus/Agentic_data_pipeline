"""Tests for auto-labeling and quality checks in the annotation stage."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.annotation_agent import AnnotationAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
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
        request=RequestConfig(topic="fitness supplements", domain="supplements", modality="text"),
    )
    return PipelineContext.from_config(config)


def _make_context_with_effect_labels(tmp_path: Path, effect_labels: list[str]) -> PipelineContext:
    """Build a minimal pipeline context with a custom effect-label vocabulary."""

    config = AppConfig(
        project=ProjectConfig(name="minecraft-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(confidence_threshold=0.6, effect_labels=effect_labels),
        request=RequestConfig(topic="minecraft instructions", domain="minecraft", modality="text"),
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


def test_annotation_prompt_contract_is_russian_and_structured(tmp_path: Path) -> None:
    """The annotation prompt should stay narrow and explicit for future LLM providers."""

    agent = AnnotationAgent(_make_context(tmp_path), registry=FakeRegistry())

    prompt = agent.build_annotation_prompt("Пример отзыва", ["energy", "side_effects", "other"])

    assert "Верни только JSON" in prompt
    assert "effect_label" in prompt
    assert "sentiment_label" in prompt
    assert "confidence" in prompt
    assert "negative, neutral или positive" in prompt
    assert "energy, side_effects, other" in prompt
    assert "Пример отзыва" in prompt
    assert "пищевых добавках" not in prompt


def test_annotation_prompt_uses_current_topic_and_avoids_fitness_only_wording(tmp_path: Path) -> None:
    """Prompt text should follow the active config topic instead of a supplements-only narrative."""

    effect_labels = ["crafting", "combat", "enchantments"]
    agent = AnnotationAgent(_make_context_with_effect_labels(tmp_path, effect_labels), registry=FakeRegistry())

    prompt = agent.build_annotation_prompt("Enchantments help with progression.", effect_labels)

    assert "minecraft instructions" in prompt
    assert "пищевых добавках" not in prompt
    assert "Текст для разметки" in prompt


def test_auto_label_parses_partial_generate_output_with_safe_fallback(tmp_path: Path) -> None:
    """A partially broken generate() response should not break the pipeline."""

    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return '{"effect_label": "energy", "sentiment_label": "positive", "confidence": "oops"}'

    agent = AnnotationAgent(_make_context(tmp_path), llm_client=FakeLLM(), registry=FakeRegistry())

    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "This supplement gives me more energy.", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    rows = labeled.to_dict(orient="records")
    trace = agent.get_annotation_trace()

    assert rows[0]["effect_label"] == "energy"
    assert rows[0]["sentiment_label"] == "positive"
    assert rows[0]["confidence"] == 0.5
    assert trace["llm_mode"] == "generate_parse"
    assert trace["n_rows"] == 1
    assert trace["n_fallback_rows"] == 1
    assert trace["parser_contract"]["parse_status_counts"]["partial_fallback"] == 1
    assert "confidence" in trace["parser_contract"]["fallback_reason_counts"]


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
    assert report["agreement"] is None
    assert report["kappa"] is None


def test_check_quality_computes_agreement_and_kappa_from_reviewed_effect_labels(tmp_path: Path) -> None:
    """Quality summaries should compare auto effect labels against reviewed_effect_label when present."""

    agent = AnnotationAgent(_make_context(tmp_path), registry=FakeRegistry())

    report = agent.check_quality(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "Crafting guide", "label": None, "effect_label": "crafting", "reviewed_effect_label": "crafting", "confidence": 0.9, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "2", "source": "HF", "text": "Combat tips", "label": None, "effect_label": "combat", "reviewed_effect_label": "crafting", "confidence": 0.4, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "3", "source": "HF", "text": "Enchantments guide", "label": None, "effect_label": "crafting", "reviewed_effect_label": "crafting", "confidence": 0.8, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    assert report["label_dist"] == {"crafting": 2 / 3, "combat": 1 / 3}
    assert report["agreement"] == pytest.approx(2 / 3)
    assert report["kappa"] is None or isinstance(report["kappa"], float)
    assert report["confidence_mean"] == pytest.approx((0.9 + 0.4 + 0.8) / 3)
    assert report["n_low_confidence"] == 1


def test_check_quality_returns_none_agreement_without_human_labels(tmp_path: Path) -> None:
    """When reviewed_effect_label is absent, agreement and kappa should remain unset."""

    agent = AnnotationAgent(_make_context(tmp_path), registry=FakeRegistry())

    report = agent.check_quality(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "Crafting guide", "label": None, "effect_label": "crafting", "confidence": 0.9, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "2", "source": "HF", "text": "Combat tips", "label": None, "effect_label": "combat", "confidence": 0.4, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    assert report["agreement"] is None
    assert report["kappa"] is None


def test_check_quality_uses_effect_labels_when_present(tmp_path: Path) -> None:
    """Quality summaries should prefer effect_label and ignore conflicting label values."""

    agent = AnnotationAgent(_make_context(tmp_path), registry=FakeRegistry())

    report = agent.check_quality(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "Crafting instructions", "label": "positive", "effect_label": "crafting", "confidence": 0.9, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "2", "source": "HF", "text": "Combat tips", "label": "negative", "effect_label": "combat", "confidence": 0.4, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "3", "source": "HF", "text": "Enchantments guide", "label": "neutral", "effect_label": "crafting", "confidence": 0.8, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    assert report["label_dist"] == {"crafting": 2 / 3, "combat": 1 / 3}
    assert report["n_rows"] == 3
    assert report["confidence_mean"] == pytest.approx((0.9 + 0.4 + 0.8) / 3)
    assert report["n_low_confidence"] == 1


def test_check_quality_uses_configured_default_label_when_other_is_absent(tmp_path: Path) -> None:
    """When 'other' is missing, label_dist should fall back to the configured default label."""

    effect_labels = ["crafting", "combat", "enchantments"]
    agent = AnnotationAgent(_make_context_with_effect_labels(tmp_path, effect_labels), registry=FakeRegistry())

    report = agent.check_quality(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "Crafting guide", "label": None, "effect_label": "crafting", "confidence": 0.9, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "2", "source": "HF", "text": "Missing effect label", "label": None, "effect_label": "", "confidence": 0.4, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
                {"id": "3", "source": "HF", "text": "Another missing effect label", "label": None, "effect_label": None, "confidence": 0.8, "rating": None, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    assert "other" not in report["label_dist"]
    assert report["label_dist"] == {"crafting": 1.0}
    assert report["agreement"] is None
    assert report["kappa"] is None


def test_check_quality_empty_input_is_safe(tmp_path: Path) -> None:
    """Empty labeled input should return a safe zeroed summary."""

    agent = AnnotationAgent(_make_context(tmp_path), registry=FakeRegistry())

    report = agent.check_quality(_Frame([]))

    assert report == {
        "label_dist": {},
        "confidence_mean": 0.0,
        "n_low_confidence": 0,
        "n_rows": 0,
        "confidence_threshold": 0.6,
        "agreement": None,
        "kappa": None,
    }


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
