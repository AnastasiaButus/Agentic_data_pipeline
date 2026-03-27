"""Tests for explicit runtime-mode selection and reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RuntimeConfig, SourceConfig
from src.core.runtime import build_runtime_summary, infer_runtime_mode, validate_runtime_mode


def _make_config(
    tmp_path: Path,
    *,
    project_name: str = "custom-project",
    source: SourceConfig | None = None,
    runtime: RuntimeConfig | None = None,
) -> AppConfig:
    """Build a minimal app config for runtime tests."""

    return AppConfig(
        project=ProjectConfig(name=project_name, root_dir=tmp_path),
        source=source if source is not None else SourceConfig(),
        annotation=AnnotationConfig(),
        runtime=runtime if runtime is not None else RuntimeConfig(),
    )


def test_infer_runtime_mode_uses_offline_demo_for_known_demo_configs(tmp_path: Path) -> None:
    """Known demo configs should default to offline_demo when runtime.mode is omitted."""

    config = _make_config(
        tmp_path,
        project_name="universal-agentic-data-pipeline-fitness-demo",
        source=SourceConfig(use_huggingface=True),
    )

    assert infer_runtime_mode(config) == "offline_demo"


def test_infer_runtime_mode_uses_online_for_non_demo_remote_configs(tmp_path: Path) -> None:
    """Remote-source configs should default to online mode outside the built-in demos."""

    config = _make_config(
        tmp_path,
        source=SourceConfig(use_huggingface=True, use_github_search=True),
    )

    assert infer_runtime_mode(config) == "online"


def test_infer_runtime_mode_uses_local_only_without_demo_or_remote_flags(tmp_path: Path) -> None:
    """Configs without demo identity or remote flags should stay fully local."""

    config = _make_config(tmp_path)

    assert infer_runtime_mode(config) == "local_only"


def test_explicit_runtime_mode_overrides_demo_default(tmp_path: Path) -> None:
    """An explicit runtime.mode should take precedence over project-name inference."""

    config = _make_config(
        tmp_path,
        project_name="universal-agentic-data-pipeline-fitness-demo",
        source=SourceConfig(use_huggingface=True),
        runtime=RuntimeConfig(mode="hybrid"),
    )

    assert infer_runtime_mode(config) == "hybrid"


def test_build_runtime_summary_distinguishes_configured_vs_active_remote_sources(tmp_path: Path) -> None:
    """Runtime summary should show when remote flags are configured but inactive in offline_demo."""

    config = _make_config(
        tmp_path,
        project_name="universal-agentic-data-pipeline-fitness-demo",
        source=SourceConfig(use_huggingface=True, use_github_search=True),
    )

    summary = build_runtime_summary(config)

    assert summary["requested_mode"] == "auto"
    assert summary["effective_mode"] == "offline_demo"
    assert summary["demo_key"] == "fitness"
    assert summary["demo_sources_enabled"] is True
    assert summary["remote_sources_enabled"] is False
    assert summary["configured_remote_source_types"] == ["hf_dataset", "github_repo"]
    assert summary["active_remote_source_types"] == []


def test_validate_runtime_mode_normalizes_hyphenated_values() -> None:
    """Hyphenated runtime values from YAML or CLI-like inputs should normalize cleanly."""

    assert validate_runtime_mode("local-only") == "local_only"


def test_validate_runtime_mode_rejects_unknown_values() -> None:
    """Unsupported runtime.mode values should fail fast."""

    with pytest.raises(ValueError, match="Unsupported runtime.mode"):
        validate_runtime_mode("spaceship")
