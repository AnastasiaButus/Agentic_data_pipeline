"""Integration coverage for the quality stage and artifact persistence."""

from __future__ import annotations

from pathlib import Path

from src.agents.data_quality_agent import DataQualityAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext


class StubRegistry:
    """Record saved artifacts without touching the filesystem."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, object]] = []

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        self.saved.append((str(path), df))
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
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def test_quality_pipeline_saves_cleaned_and_compare_artifacts(tmp_path: Path) -> None:
    """The quality stage should persist cleaned and comparison artifacts without using the network."""

    agent = DataQualityAgent(_make_context(tmp_path), registry=StubRegistry())
    dirty = _Frame(
        [
            {"id": "1", "source": "HF", "text": "  Great product  ", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "2", "source": "HF", "text": "Great product", "label": "positive", "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "3", "source": "Web", "text": "", "label": "negative", "rating": 2, "created_at": "now", "split": None, "meta_json": "{}"},
            {"id": "4", "source": "Web", "text": " ".join(["long"] * 50), "label": "negative", "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        ]
    )

    result = agent.run(
        dirty,
        {
            "drop_empty_text": True,
            "min_words": 1,
            "normalize_whitespace": True,
            "duplicates": "drop",
            "outliers": "remove_iqr",
        },
    )

    saved_paths = [path for path, _ in agent.registry.saved]
    compare_rows = agent.registry.saved[1][1].to_dict(orient="records")

    assert list(result.columns) == ["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"]
    assert saved_paths == ["data/interim/cleaned_v1.parquet", "data/interim/quality_compare.csv"]
    assert compare_rows[0]["metric"] == "n_rows"
    assert compare_rows[0]["before"] == 4.0
    assert compare_rows[0]["after"] == 2.0
    assert len(result.to_dict(orient="records")) == 2


def test_quality_agent_is_compatible_with_base_agent(tmp_path: Path) -> None:
    """The quality agent should still work with the shared BaseAgent contract."""

    agent = DataQualityAgent(_make_context(tmp_path), registry=StubRegistry())

    assert agent.name == "DataQualityAgent"