"""Configuration models and loading helpers for the pipeline."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any

from .constants import DEFAULT_RANDOM_SEED
from .exceptions import ConfigError


@dataclass(slots=True)
class ProjectConfig:
    """Static metadata about the project and its execution root."""

    name: str
    root_dir: Path = Path(".")
    domain: str = "fitness_supplements"


@dataclass(slots=True)
class RequestConfig:
    """Parameters for external requests or retrieval operations."""

    timeout_seconds: int = 30
    max_retries: int = 3
    batch_size: int = 32


@dataclass(slots=True)
class SourceConfig:
    """Settings that describe where input data comes from."""

    name: str
    input_path: Path | None = None
    enabled: bool = True
    allowed_extensions: tuple[str, ...] = (".csv", ".json", ".jsonl", ".txt")


@dataclass(slots=True)
class AnnotationConfig:
    """Labels and field names used during annotation."""

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
    source: SourceConfig
    annotation: AnnotationConfig
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
    raw_config = _parse_yaml_subset(text)
    if not raw_config:
        raise ConfigError("Configuration file is empty")
    if not isinstance(raw_config, dict):
        raise ConfigError("Configuration root must be a mapping")

    project_data = _require_section(raw_config, "project")
    source_data = _require_section(raw_config, "source")
    annotation_data = _require_section(raw_config, "annotation")

    return AppConfig(
        project=_build_config(
            ProjectConfig,
            project_data,
            path_fields={"root_dir"},
            required_fields={"name"},
        ),
        source=_build_config(
            SourceConfig,
            source_data,
            path_fields={"input_path"},
            required_fields={"name"},
        ),
        annotation=_build_config(
            AnnotationConfig,
            annotation_data,
            path_fields={"output_dir"},
        ),
        request=_build_config(RequestConfig, raw_config.get("request", {})),
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
        elif field_info.name == "allowed_extensions" and value is not None:
            value = tuple(value)
        kwargs[field_info.name] = value

    # Ignore unknown keys so forward-compatible configs do not break this step.
    return config_type(**kwargs)


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    lines = _normalize_lines(text)
    if not lines:
        return {}

    value, index = _parse_block(lines, 0, lines[0][0])
    while index < len(lines):
        if lines[index][1]:
            raise ConfigError("Unexpected trailing content in configuration file")
        index += 1
    if not isinstance(value, dict):
        raise ConfigError("Configuration root must be a mapping")
    return value


def _normalize_lines(text: str) -> list[tuple[int, str]]:
    normalized: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        normalized.append((indent, stripped.lstrip(" ")))
    return normalized


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    current_indent, current_text = lines[index]
    if current_text.startswith("- ") and current_indent == indent:
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        current_indent, current_text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ConfigError("Invalid indentation in configuration file")
        if current_text.startswith("- "):
            break

        key, separator, remainder = current_text.partition(":")
        if not separator:
            raise ConfigError(f"Invalid mapping entry: {current_text}")

        key = key.strip()
        remainder = remainder.strip()
        index += 1

        if remainder:
            result[key] = _parse_scalar(remainder)
            continue

        if index >= len(lines) or lines[index][0] <= indent:
            result[key] = {}
            continue

        child_indent = lines[index][0]
        child_value, index = _parse_block(lines, index, child_indent)
        result[key] = child_value

    return result, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        current_indent, current_text = lines[index]
        if current_indent < indent or not current_text.startswith("- "):
            break
        if current_indent > indent:
            raise ConfigError("Invalid indentation inside a list")

        item_text = current_text[2:].strip()
        index += 1

        if item_text:
            result.append(_parse_scalar(item_text))
            continue

        if index >= len(lines) or lines[index][0] <= indent:
            result.append(None)
            continue

        child_indent = lines[index][0]
        child_value, index = _parse_block(lines, index, child_indent)
        result.append(child_value)

    return result, index


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value[:1] in {"[", "{", '"', "'"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value