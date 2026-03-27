"""Configuration models and loading helpers for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_RANDOM_SEED
from .exceptions import ConfigError


@dataclass(slots=True)
class ProjectConfig:
    """Static metadata about the project and its execution root."""

    name: str
    root_dir: Path = Path(".")
    seed: int = DEFAULT_RANDOM_SEED


@dataclass(slots=True)
class RequestConfig:
    """Request planning and source-selection defaults for the current task."""

    topic: str = ""
    modality: str = ""
    task_type: str = ""
    domain: str = ""
    sources_preference: list[str] = field(default_factory=list)
    label_schema: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceConfig:
    """Source selection switches for the ingestion layer."""

    use_huggingface: bool = False
    use_public_api: bool = False
    use_internal_api: bool = False
    use_github_search: bool = False
    use_scraping_fallback: bool = False
    max_sources: int = 0


@dataclass(slots=True)
class AnnotationConfig:
    """Annotation defaults, provider selection, and output field names for review labeling."""

    use_llm: bool = False
    llm_provider: str = ""
    confidence_threshold: float = 0.0
    effect_labels: list[str] = field(default_factory=list)
    text_field: str = "text"
    label_field: str = "label"
    id_field: str = "id"
    output_dir: Path | None = None


@dataclass(slots=True)
class QualityConfig:
    """Basic quality gates for cleaning and filtering records."""

    min_text_length: int = 20
    deduplicate: bool = True
    max_missing_ratio: float = 0.2


@dataclass(slots=True)
class ActiveLearningConfig:
    """Settings for the active-learning loop."""

    enabled: bool = False
    query_size: int = 32
    uncertainty_threshold: float = 0.2


@dataclass(slots=True)
class TrainingConfig:
    """Training defaults for downstream model experiments."""

    enabled: bool = False
    random_seed: int = DEFAULT_RANDOM_SEED
    test_size: float = 0.2
    validation_size: float = 0.1


@dataclass(slots=True)
class AppConfig:
    """Aggregate configuration for the full pipeline runtime."""

    project: ProjectConfig
    source: SourceConfig = field(default_factory=SourceConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load an application config from a YAML file.

    Unknown keys are ignored, but required sections must be present.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    raw_config = yaml.safe_load(text)
    if not raw_config:
        raise ConfigError("Configuration file is empty")
    if not isinstance(raw_config, dict):
        raise ConfigError("Configuration root must be a mapping")

    project_data = _require_section(raw_config, "project")
    request_data = _require_section(raw_config, "request")
    source_data = _require_section(raw_config, "source")
    annotation_data = _require_section(raw_config, "annotation")

    return AppConfig(
        project=_build_config(
            ProjectConfig,
            project_data,
            path_fields={"root_dir"},
            required_fields={"name"},
        ),
        request=_build_config(RequestConfig, request_data),
        source=_build_config(SourceConfig, source_data),
        annotation=_build_config(
            AnnotationConfig,
            annotation_data,
            path_fields={"output_dir"},
        ),
        quality=_build_config(QualityConfig, raw_config.get("quality", {})),
        active_learning=_build_config(
            ActiveLearningConfig,
            raw_config.get("active_learning", {}),
        ),
        training=_build_config(TrainingConfig, raw_config.get("training", {})),
    )


def _require_section(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = raw_config.get(section_name)
    if section is None:
        raise ConfigError(f"Missing required section: {section_name}")
    if not isinstance(section, dict):
        raise ConfigError(f"Section '{section_name}' must be a mapping")
    return section


def _build_config(
    config_type: type[Any],
    values: dict[str, Any] | None,
    *,
    path_fields: set[str] | None = None,
    required_fields: set[str] | None = None,
) -> Any:
    data = values or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Section for {config_type.__name__} must be a mapping")

    path_fields = path_fields or set()
    required_fields = required_fields or set()

    missing = [name for name in required_fields if name not in data]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ConfigError(f"Missing required field(s) for {config_type.__name__}: {missing_list}")

    kwargs: dict[str, Any] = {}
    for field_info in fields(config_type):
        if field_info.name not in data:
            continue
        value = data[field_info.name]
        if field_info.name in path_fields and value is not None:
            value = Path(value)
        elif field_info.name in {"sources_preference", "effect_labels"} and value is not None:
            value = list(value)
        kwargs[field_info.name] = value

    # Ignore unknown keys so forward-compatible configs do not break this step.
    return config_type(**kwargs)
