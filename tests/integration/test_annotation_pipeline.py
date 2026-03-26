"""Integration coverage for the annotation stage and Label Studio export."""

from __future__ import annotations

from pathlib import Path

from src.agents.annotation_agent import AnnotationAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.providers.labelstudio.exporter import to_labelstudio_tasks
from src.providers.labelstudio.validators import validate_labelstudio_tasks
from src.providers.llm.mock_llm import MockLLM


class FakeRegistry:
    """Capture artifact writes without touching the filesystem."""

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        return Path(path)


class _Frame:
    """Tiny dataframe-like helper used to keep the integration test independent of pandas."""

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
    """Build a minimal pipeline context for the integration test."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(confidence_threshold=0.6),
    )
    return PipelineContext.from_config(config)


def test_annotation_pipeline_accepts_canonical_schema_and_exports_labelstudio_tasks(tmp_path: Path) -> None:
    """The annotation stage should stay on the canonical schema and produce review-ready exports."""

    agent = AnnotationAgent(_make_context(tmp_path), llm_client=MockLLM(), registry=FakeRegistry())
    canonical_df = _Frame(
        [
            {"id": "1", "source": "HF", "text": "This supplement gives me more energy.", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "2", "source": "Web", "text": "I noticed a side effect after a week.", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        ]
    )

    labeled = agent.auto_label(canonical_df)
    quality = agent.check_quality(labeled)
    tasks = agent.export_to_labelstudio(labeled)
    helper_tasks = to_labelstudio_tasks(labeled)

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
    assert quality["n_rows"] == 2
    validate_labelstudio_tasks(tasks)
    assert tasks[0]["data"]["text"] == "This supplement gives me more energy."
    assert tasks[0]["predictions"][0]["result"]
    assert tasks == helper_tasks