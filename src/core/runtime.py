"""Helpers for explicit pipeline runtime modes and source activation."""

from __future__ import annotations

from typing import Any

from src.core.config import AppConfig


SUPPORTED_RUNTIME_MODES = {"offline_demo", "online", "hybrid", "local_only"}
DEMO_PROJECT_NAME_TO_KEY = {
    "universal-agentic-data-pipeline-fitness-demo": "fitness",
    "universal-agentic-data-pipeline-minecraft-demo": "minecraft",
}


def normalize_runtime_mode(value: Any) -> str:
    """Normalize runtime mode values from config and CLI-like inputs."""

    return str(value or "").strip().lower().replace("-", "_")


def validate_runtime_mode(value: Any) -> str:
    """Validate the configured runtime mode and return the normalized value."""

    normalized = normalize_runtime_mode(value)
    if normalized and normalized not in SUPPORTED_RUNTIME_MODES:
        supported = ", ".join(sorted(SUPPORTED_RUNTIME_MODES))
        raise ValueError(f"Unsupported runtime.mode='{value}'. Supported values: {supported}")
    return normalized


def demo_key_for_config(config: AppConfig) -> str | None:
    """Return the built-in demo key for known project identities."""

    project_name = str(getattr(getattr(config, "project", None), "name", "") or "").strip()
    return DEMO_PROJECT_NAME_TO_KEY.get(project_name)


def infer_runtime_mode(config: AppConfig) -> str:
    """Resolve the effective runtime mode with backward-compatible defaults."""

    requested_mode = validate_runtime_mode(getattr(getattr(config, "runtime", None), "mode", ""))
    if requested_mode:
        return requested_mode

    if demo_key_for_config(config) is not None:
        return "offline_demo"

    if remote_sources_requested(config):
        return "online"

    return "local_only"


def runtime_allows_demo_sources(config: AppConfig) -> bool:
    """Return whether the effective mode can use built-in local/demo candidates."""

    return infer_runtime_mode(config) in {"offline_demo", "hybrid", "local_only"} and demo_key_for_config(config) is not None


def runtime_allows_remote_sources(config: AppConfig) -> bool:
    """Return whether the effective mode can call remote discovery providers."""

    return infer_runtime_mode(config) in {"online", "hybrid"}


def remote_sources_requested(config: AppConfig) -> bool:
    """Return whether any remote source flag is enabled in config."""

    source = getattr(config, "source", None)
    if source is None:
        return False

    return any(
        bool(getattr(source, field_name, False))
        for field_name in (
            "use_huggingface",
            "use_public_api",
            "use_internal_api",
            "use_github_search",
            "use_scraping_fallback",
        )
    )


def configured_remote_source_types(config: AppConfig) -> list[str]:
    """Return the remote source kinds explicitly requested by config flags."""

    source = getattr(config, "source", None)
    if source is None:
        return []

    flags = [
        ("hf_dataset", "use_huggingface"),
        ("internal_api", "use_internal_api"),
        ("public_api", "use_public_api"),
        ("github_repo", "use_github_search"),
        ("scrape", "use_scraping_fallback"),
    ]
    return [source_type for source_type, field_name in flags if bool(getattr(source, field_name, False))]


def active_remote_source_types(config: AppConfig) -> list[str]:
    """Return the remote source kinds that are active in the effective runtime."""

    if not runtime_allows_remote_sources(config):
        return []
    return configured_remote_source_types(config)


def build_runtime_summary(config: AppConfig) -> dict[str, Any]:
    """Build a compact runtime summary for reports and UI surfaces."""

    effective_mode = infer_runtime_mode(config)
    requested_mode = validate_runtime_mode(getattr(getattr(config, "runtime", None), "mode", "")) or "auto"
    demo_key = demo_key_for_config(config)

    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "demo_key": demo_key or "",
        "demo_sources_enabled": runtime_allows_demo_sources(config),
        "remote_sources_enabled": runtime_allows_remote_sources(config),
        "configured_remote_source_types": configured_remote_source_types(config),
        "active_remote_source_types": active_remote_source_types(config),
    }
