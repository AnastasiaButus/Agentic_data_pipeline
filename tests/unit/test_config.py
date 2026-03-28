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
runtime:
  mode: hybrid
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
        assert config.runtime.mode == "hybrid"
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
runtime:
  mode: local_only
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
    assert config.runtime.mode == "local_only"


def test_load_config_rejects_invalid_runtime_mode(tmp_path: Path) -> None:
    """Unsupported runtime modes should fail fast during config loading."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: fitness-demo
request:
  topic: fitness supplements
source:
  use_huggingface: true
annotation:
  use_llm: false
runtime:
  mode: spaceship
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="runtime.mode"):
        load_config(config_path)


def test_text_topic_template_config_loads_from_repo() -> None:
    """The generic text-topic template should stay aligned with the supported config contract."""

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "text_topic_template.yaml"

    config = load_config(config_path)

    assert config.project.name == "universal-agentic-data-pipeline-text-topic"
    assert config.request.modality == "text"
    assert config.request.task_type == "classification"
    assert config.source.use_huggingface is True
    assert config.annotation.use_llm is False
    assert config.annotation.effect_labels == ["class_one", "class_two", "other"]
    assert config.runtime.mode == "online"


def test_text_topic_online_config_loads_from_repo() -> None:
    """The dedicated online text-topic config should expose the intended remote source flags."""

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "text_topic_online.yaml"

    config = load_config(config_path)

    assert config.project.name == "universal-agentic-data-pipeline-text-topic-online"
    assert config.request.modality == "text"
    assert config.source.use_huggingface is True
    assert config.source.use_github_search is True
    assert config.source.use_scraping_fallback is False
    assert config.runtime.mode == "online"


def test_text_topic_hybrid_config_loads_from_repo() -> None:
    """The dedicated hybrid text-topic config should preserve the mixed-mode runtime contract."""

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "text_topic_hybrid.yaml"

    config = load_config(config_path)

    assert config.project.name == "universal-agentic-data-pipeline-text-topic-hybrid"
    assert config.request.modality == "text"
    assert config.source.use_huggingface is True
    assert config.source.use_github_search is True
    assert config.source.use_scraping_fallback is True
    assert config.runtime.mode == "hybrid"
