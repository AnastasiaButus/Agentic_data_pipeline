"""Tests for Label Studio export and validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_export_to_labelstudio_creates_valid_task_structure(tmp_path: Path) -> None:
    """The exported tasks should contain data and predictions in a Label Studio-friendly shape."""

    agent = AnnotationAgent(_make_context(tmp_path), llm_client=MockLLM(), registry=FakeRegistry())
    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "This supplement gives me more energy.", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    tasks = agent.export_to_labelstudio(labeled)
    helper_tasks = to_labelstudio_tasks(labeled)

    assert isinstance(tasks, list)
    assert tasks and "data" in tasks[0] and "predictions" in tasks[0]
    assert tasks[0]["data"]["id"] == "1"
    assert tasks[0]["data"]["text"] == "This supplement gives me more energy."
    assert tasks[0]["predictions"][0]["result"]
    assert tasks == helper_tasks


def test_exporter_and_validator_reject_task_without_data() -> None:
    """The validator should fail fast when the task payload is missing the required data section."""

    with pytest.raises(ValueError, match="missing data"):
        validate_labelstudio_tasks([{"predictions": []}])


def test_module_exporter_matches_agent_output(tmp_path: Path) -> None:
    """The standalone exporter should produce the same minimal task structure as the agent helper."""

    agent = AnnotationAgent(_make_context(tmp_path), llm_client=MockLLM(), registry=FakeRegistry())
    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "2", "source": "Web", "text": "I noticed a side effect after a week.", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    tasks = to_labelstudio_tasks(labeled)
    agent_tasks = agent.export_to_labelstudio(labeled)

    validate_labelstudio_tasks(tasks)

    assert tasks[0]["data"]["id"] == "2"
    assert tasks[0]["predictions"][0]["score"] >= 0.0
    assert tasks == agent_tasks