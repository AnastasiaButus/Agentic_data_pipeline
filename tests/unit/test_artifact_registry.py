"""Tests for the filesystem-backed artifact registry and IO helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.services.artifact_registry import ArtifactRegistry


class SmallTable:
    """Tiny dataframe-like helper used to keep tests independent of pandas."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        """Return rows in records orientation, matching the registry contract."""

        if orient != "records":
            raise ValueError("SmallTable only supports records orientation")
        return list(self._rows)


def _make_registry(tmp_path: Path) -> ArtifactRegistry:
    """Build a registry rooted at a temporary project directory."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    context = PipelineContext.from_config(config)
    return ArtifactRegistry(context)


def test_save_json_load_json_roundtrip(tmp_path: Path) -> None:
    """JSON payloads should round-trip through the registry."""

    registry = _make_registry(tmp_path)
    payload = {"name": "fitness-demo", "enabled": True, "threshold": None}

    saved_path = registry.save_json("artifacts/meta/payload.json", payload)

    assert saved_path.exists()
    assert registry.exists("artifacts/meta/payload.json") is True
    assert registry.load_json(Path("artifacts/meta/payload.json")) == payload


def test_save_markdown_creates_file_and_saves_text(tmp_path: Path) -> None:
    """Markdown files should be written as plain text under nested directories."""

    registry = _make_registry(tmp_path)
    markdown = "# Fitness Demo\n\n- supplements\n- reviews"

    saved_path = registry.save_markdown("reports/specs/annotation.md", markdown)

    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == markdown


def test_save_dataframe_load_dataframe_roundtrip(tmp_path: Path) -> None:
    """Tabular data should round-trip with parquet when available and CSV fallback otherwise."""

    registry = _make_registry(tmp_path)
    rows = [
        {"id": 1, "text": "good product", "label": "positive"},
        {"id": 2, "text": "too expensive", "label": "negative"},
    ]
    frame = SmallTable(rows)

    saved_path = registry.save_dataframe("data/interim/reviews.parquet", frame)
    loaded = registry.load_dataframe("data/interim/reviews.parquet")

    assert saved_path.exists()
    assert registry.exists("data/interim/reviews.parquet") is True
    if hasattr(loaded, "to_dict"):
        assert loaded.to_dict(orient="records") == rows
    else:
        assert loaded == rows


def test_missing_loads_raise_clear_errors(tmp_path: Path) -> None:
    """Missing artifacts should raise a consistent, testable error."""

    registry = _make_registry(tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.load_json("missing/payload.json")

    with pytest.raises(FileNotFoundError):
        registry.load_text("missing/spec.md")

    with pytest.raises(FileNotFoundError):
        registry.load_dataframe("missing/data.parquet")


def test_nested_directories_are_created_automatically(tmp_path: Path) -> None:
    """Saving artifacts should create intermediate directories without extra setup."""

    registry = _make_registry(tmp_path)

    registry.save_text("nested/a/b/output.txt", "hello")

    assert (tmp_path / "nested" / "a" / "b" / "output.txt").exists()


def test_resolve_handles_relative_paths_under_root(tmp_path: Path) -> None:
    """Relative artifact paths should resolve under the project root."""

    registry = _make_registry(tmp_path)

    resolved_path = registry._resolve("reports/summary.md")

    assert resolved_path == tmp_path / "reports" / "summary.md"


def test_resolve_blocks_path_traversal_outside_root(tmp_path: Path) -> None:
    """Path traversal attempts should fail instead of escaping the project root."""

    registry = _make_registry(tmp_path)

    with pytest.raises(ValueError, match="escapes project root"):
        registry._resolve("../outside.txt")


def test_resolve_accepts_absolute_paths_inside_root(tmp_path: Path) -> None:
    """Absolute paths already inside the root should resolve predictably to the same artifact."""

    registry = _make_registry(tmp_path)
    absolute_path = tmp_path / "data" / "interim" / "artifact.json"

    resolved_path = registry._resolve(absolute_path)

    assert resolved_path == absolute_path.resolve(strict=False)
