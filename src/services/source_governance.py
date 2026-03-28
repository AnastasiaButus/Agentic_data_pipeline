"""Operational governance helpers for remote source paths and fallback reporting."""

from __future__ import annotations

import os
from typing import Any

from src.core.config import AppConfig
from src.core.runtime import build_runtime_summary


def build_online_governance_summary(config: AppConfig, candidates: list[Any]) -> dict[str, Any]:
    """Summarize remote-provider activation, rate-limit awareness, and fallback behavior."""

    runtime = build_runtime_summary(config)
    counts = _count_candidates_by_provider(candidates)
    github_enabled = bool(getattr(config.source, "use_github_search", False))
    github_auth_mode = (
        "token"
        if github_enabled and _normalized_env("GITHUB_TOKEN")
        else ("unauthenticated" if github_enabled else "not_used")
    )

    providers = [
        _build_provider_row(
            provider_id="huggingface_datasets",
            label="Hugging Face datasets API",
            enabled_in_config=bool(getattr(config.source, "use_huggingface", False)),
            active_in_runtime="hf_dataset" in runtime.get("active_remote_source_types", []),
            discovered_candidates=counts.get("huggingface_datasets", 0),
            auth_mode="public",
            implementation_status="real_lookup_mvp",
            rate_limit_guidance="Use low-volume public dataset search requests; provider-side limits may still change.",
            fallback_behavior="If lookup returns no candidates, the pipeline keeps running with an empty remote shortlist.",
            operator_action="",
        ),
        _build_provider_row(
            provider_id="github_repository_search",
            label="GitHub repository search API",
            enabled_in_config=github_enabled,
            active_in_runtime="github_repo" in runtime.get("active_remote_source_types", []),
            discovered_candidates=counts.get("github_repository_search", 0),
            auth_mode=github_auth_mode,
            implementation_status="real_lookup_mvp",
            rate_limit_guidance=(
                "GitHub Search API is more fragile without GITHUB_TOKEN; authenticated requests are more stable but still quota-bound."
                if github_auth_mode == "unauthenticated"
                else "Authenticated GitHub Search is still quota-bound, but less fragile than the unauthenticated path."
            ),
            fallback_behavior="If GitHub search returns nothing, the pipeline continues without breaking the offline-first baseline.",
            operator_action=(
                "Set GITHUB_TOKEN to reduce GitHub Search API rate-limit risk."
                if bool(getattr(config.source, "use_github_search", False)) and "github_repo" in runtime.get("active_remote_source_types", []) and github_auth_mode == "unauthenticated"
                else ""
            ),
        ),
        _build_provider_row(
            provider_id="internal_api_candidate",
            label="Internal API discovery candidate",
            enabled_in_config=bool(getattr(config.source, "use_internal_api", False)),
            active_in_runtime="internal_api" in runtime.get("active_remote_source_types", []),
            discovered_candidates=counts.get("internal_api_candidate", 0),
            auth_mode="restricted_review",
            implementation_status="demo_stub_candidate",
            rate_limit_guidance="Quota, access policy, and SLA are project-specific and require human review.",
            fallback_behavior="This path only contributes approval-aware candidates; the pipeline stays runnable without it.",
            operator_action=(
                "Review access policy and implementation scope before relying on internal API discovery."
                if bool(getattr(config.source, "use_internal_api", False)) and "internal_api" in runtime.get("active_remote_source_types", [])
                else ""
            ),
        ),
        _build_provider_row(
            provider_id="public_api_candidate",
            label="Public API discovery candidate",
            enabled_in_config=bool(getattr(config.source, "use_public_api", False)),
            active_in_runtime="public_api" in runtime.get("active_remote_source_types", []),
            discovered_candidates=counts.get("public_api_candidate", 0),
            auth_mode="provider_specific",
            implementation_status="demo_stub_candidate",
            rate_limit_guidance="Quota and terms depend on the external provider and should be reviewed before production use.",
            fallback_behavior="This path only contributes approval-aware candidates; the pipeline keeps running without it.",
            operator_action=(
                "Review provider quota and terms before promoting public API discovery beyond MVP."
                if bool(getattr(config.source, "use_public_api", False)) and "public_api" in runtime.get("active_remote_source_types", [])
                else ""
            ),
        ),
        _build_provider_row(
            provider_id="scraping_fallback_candidate",
            label="Scraping fallback candidate",
            enabled_in_config=bool(getattr(config.source, "use_scraping_fallback", False)),
            active_in_runtime="scrape" in runtime.get("active_remote_source_types", []),
            discovered_candidates=counts.get("scraping_fallback_candidate", 0),
            auth_mode="not_applicable",
            implementation_status="candidate_stub",
            rate_limit_guidance="Respect site-specific limits and robots.txt before turning scraping into a real online path.",
            fallback_behavior="When scraping is unavailable, the pipeline keeps running and preserves the offline/demo baseline.",
            operator_action=(
                "Keep scraping behind human approval and robots.txt review."
                if bool(getattr(config.source, "use_scraping_fallback", False)) and "scrape" in runtime.get("active_remote_source_types", [])
                else ""
            ),
        ),
    ]

    notes: list[str] = []
    configured_remote = runtime.get("configured_remote_source_types", [])
    if configured_remote and not bool(runtime.get("remote_sources_enabled")):
        notes.append("Remote source flags are configured, but current runtime.mode keeps them inactive.")
    if any(row["observed_status"] == "active_no_candidates" for row in providers):
        notes.append("At least one active remote provider returned no candidates, so the pipeline continued with a fallback-safe empty remote shortlist.")
    if any(row["provider_id"] == "github_repository_search" and row["operator_action"] for row in providers):
        notes.append("Configure GITHUB_TOKEN to make the GitHub discovery path less fragile under Search API rate limits.")

    return {
        "remote_sources_enabled": bool(runtime.get("remote_sources_enabled")),
        "configured_remote_source_types": list(configured_remote) if isinstance(configured_remote, list) else [],
        "active_remote_source_types": list(runtime.get("active_remote_source_types", [])) if isinstance(runtime.get("active_remote_source_types", []), list) else [],
        "active_provider_count": sum(1 for row in providers if row["active_in_runtime"]),
        "providers_requiring_attention": [row["provider_id"] for row in providers if row["operator_action"]],
        "github_auth_mode": github_auth_mode,
        "fallback_strategy": "empty remote shortlist keeps the run stable and preserves the offline-first baseline",
        "providers": providers,
        "notes": notes,
    }


def _build_provider_row(
    *,
    provider_id: str,
    label: str,
    enabled_in_config: bool,
    active_in_runtime: bool,
    discovered_candidates: int,
    auth_mode: str,
    implementation_status: str,
    rate_limit_guidance: str,
    fallback_behavior: str,
    operator_action: str,
) -> dict[str, Any]:
    """Build one provider row for the governance summary."""

    observed_status = _provider_observed_status(
        enabled_in_config=enabled_in_config,
        active_in_runtime=active_in_runtime,
        discovered_candidates=discovered_candidates,
    )
    return {
        "provider_id": provider_id,
        "label": label,
        "enabled_in_config": enabled_in_config,
        "active_in_runtime": active_in_runtime,
        "discovered_candidates": discovered_candidates,
        "observed_status": observed_status,
        "auth_mode": auth_mode,
        "implementation_status": implementation_status,
        "rate_limit_guidance": rate_limit_guidance,
        "fallback_behavior": fallback_behavior,
        "operator_action": operator_action,
    }


def _provider_observed_status(
    *,
    enabled_in_config: bool,
    active_in_runtime: bool,
    discovered_candidates: int,
) -> str:
    """Map provider activation and candidate counts to a stable status label."""

    if not enabled_in_config:
        return "disabled_in_config"
    if not active_in_runtime:
        return "configured_but_inactive_for_runtime"
    if discovered_candidates > 0:
        return "returned_candidates"
    return "active_no_candidates"


def _count_candidates_by_provider(candidates: list[Any]) -> dict[str, int]:
    """Count discovered candidates by governance provider id."""

    counts: dict[str, int] = {}
    for candidate in candidates:
        provider_id = _candidate_provider_id(candidate)
        if not provider_id:
            continue
        counts[provider_id] = counts.get(provider_id, 0) + 1
    return counts


def _candidate_provider_id(candidate: Any) -> str:
    """Resolve a discovered candidate to the provider bucket used for governance reporting."""

    source_type = _normalized_text(getattr(candidate, "source_type", ""))
    uri = _normalized_text(getattr(candidate, "uri", ""))
    metadata = getattr(candidate, "metadata", None)
    normalized_metadata = metadata if isinstance(metadata, dict) else {}

    if source_type == "hf_dataset":
        return "huggingface_datasets"
    if source_type == "github_repo":
        return "github_repository_search"
    if source_type == "api":
        api_kind = _normalized_text(normalized_metadata.get("api_kind")).lower()
        return "internal_api_candidate" if api_kind == "internal" else "public_api_candidate"
    if source_type == "scrape":
        demo_mode = bool(normalized_metadata.get("demo_mode")) or uri.startswith("demo://")
        if demo_mode:
            return ""
        return "scraping_fallback_candidate"
    return ""


def _normalized_env(name: str) -> str:
    """Return a stripped environment variable value."""

    return os.getenv(name, "").strip()


def _normalized_text(value: Any) -> str:
    """Normalize arbitrary values into stable strings."""

    if value is None:
        return ""
    return str(value).strip()
