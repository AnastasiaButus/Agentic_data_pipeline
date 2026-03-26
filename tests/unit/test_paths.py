"""Tests for pipeline filesystem paths and context wiring."""

from __future__ import annotations

from pathlib import Path

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.constants import STANDARD_COLUMNS
from src.core.context import PipelineContext
from src.core.paths import PipelinePaths


def test_paths_ensure_dirs_creates_expected_directories(tmp_path: Path) -> None:
    """The standard directory tree should be created and be idempotent."""

    paths = PipelinePaths(tmp_path)
    paths.ensure_dirs()
    paths.ensure_dirs()

    expected = (
        paths.data_raw,
        paths.data_interim,
        paths.data_labeled,
        paths.models,
        paths.reports,
        paths.reports_figures,
        paths.configs,
    )

    for path in expected:
        assert path.exists()
        assert path.is_dir()


def test_context_from_config_uses_tmp_root(tmp_path: Path) -> None:
    """Context creation should honor the explicit root override."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=Path("/not-used")),
        source=SourceConfig(name="reviews"),
        annotation=AnnotationConfig(),
    )

    context = PipelineContext.from_config(config, root_dir=tmp_path)

    assert context.config == config
    assert context.paths.root_dir == tmp_path
    assert context.paths.data_raw == tmp_path / "data" / "raw"


def test_standard_columns_contains_required_fields() -> None:
    """The shared column set should include the canonical review fields."""

    required = {"id", "source", "text", "label"}
    assert required.issubset(set(STANDARD_COLUMNS))