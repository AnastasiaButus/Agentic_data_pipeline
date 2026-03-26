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
  seed: 7
request:
  topic: supplements reviews
  modality: text
  task_type: classification
  domain: fitness_supplements
  sources_preference:
    - huggingface
    - internal_api
  label_schema:
    positive: 1
    negative: 0
  constraints:
    min_text_length: 20
source:
  use_huggingface: true
  use_public_api: false
  use_internal_api: true
  use_github_search: false
  use_scraping_fallback: true
  max_sources: 5
annotation:
  use_llm: false
  llm_provider: null
  confidence_threshold: 0.75
  effect_labels:
    - positive
    - negative
  text_field: review_text
  label_field: sentiment
  id_field: review_id
  output_dir: null
unknown_section:
  ignored: true
""".strip(),
        encoding="utf-8",
    )

    for path_value in (str(config_path), config_path):
        config = load_config(path_value)

        assert config.project.name == "fitness-demo"
        assert config.project.root_dir == Path(".")
        assert config.project.seed == 7
        assert config.request.topic == "supplements reviews"
        assert config.request.sources_preference == ["huggingface", "internal_api"]
        assert config.source.use_huggingface is True
        assert config.source.max_sources == 5
        assert config.annotation.label_field == "sentiment"
        assert config.annotation.text_field == "review_text"
        assert config.annotation.llm_provider is None
        assert config.training.random_seed == 42


def test_load_config_missing_required_section(tmp_path: Path) -> None:
    """Missing required sections should raise a config error."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: fitness-demo
source:
  use_huggingface: true
annotation:
  text_field: review_text
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_load_config_supports_lists_bools_and_nulls(tmp_path: Path) -> None:
    """YAML lists, booleans, and nulls should round-trip through safe_load."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: fitness-demo
  seed: 99
request:
  topic: fitness supplements
  modality: text
  task_type: extraction
  domain: fitness_supplements
  sources_preference:
    - public_api
    - scraping_fallback
  label_schema:
    labels:
      - positive
      - negative
  constraints:
    enabled: true
    ratio: 0.25
    fallback: null
source:
  use_huggingface: false
  use_public_api: true
  use_internal_api: false
  use_github_search: true
  use_scraping_fallback: false
  max_sources: 12
annotation:
  use_llm: true
  llm_provider: null
  confidence_threshold: 0.5
  effect_labels:
    - boost
    - neutral
  text_field: text
  label_field: label
  id_field: id
  output_dir: null
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project.seed == 99
    assert config.request.sources_preference == ["public_api", "scraping_fallback"]
    assert config.request.constraints == {"enabled": True, "ratio": 0.25, "fallback": None}
    assert config.source.use_public_api is True
    assert config.source.use_github_search is True
    assert config.source.max_sources == 12
    assert config.annotation.use_llm is True
    assert config.annotation.llm_provider is None
    assert config.annotation.effect_labels == ["boost", "neutral"]