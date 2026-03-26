"""Tests for core configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import load_config
from src.core.exceptions import ConfigError


def test_load_config_success(tmp_path: Path) -> None:
    """Config loading should accept both string and Path inputs."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: fitness-demo
  root_dir: .
  domain: fitness_supplements
source:
  name: reviews
annotation:
  text_field: review_text
unknown_section:
  ignored: true
""".strip(),
        encoding="utf-8",
    )

    for path_value in (str(config_path), config_path):
        config = load_config(path_value)

        assert config.project.name == "fitness-demo"
        assert config.project.root_dir == Path(".")
        assert config.source.name == "reviews"
        assert config.annotation.text_field == "review_text"
        assert config.request.batch_size == 32
        assert config.training.random_seed == 42


def test_load_config_missing_required_section(tmp_path: Path) -> None:
    """Missing required sections should raise a config error."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: fitness-demo
annotation:
  text_field: review_text
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(config_path)