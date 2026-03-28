"""Unit tests for online governance and remote-provider fallback reporting."""

from __future__ import annotations

from pathlib import Path

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RuntimeConfig, SourceConfig
from src.domain import SourceCandidate
from src.services.source_governance import build_online_governance_summary


def _make_config(
    tmp_path: Path,
    *,
    project_name: str = "custom-project",
    source: SourceConfig | None = None,
    runtime: RuntimeConfig | None = None,
) -> AppConfig:
    """Build a minimal config for governance-summary tests."""

    return AppConfig(
        project=ProjectConfig(name=project_name, root_dir=tmp_path),
        source=source if source is not None else SourceConfig(),
        annotation=AnnotationConfig(),
        runtime=runtime if runtime is not None else RuntimeConfig(),
    )


def _provider(summary: dict[str, object], provider_id: str) -> dict[str, object]:
    """Return one provider row from the governance summary."""

    providers = summary.get("providers", [])
    assert isinstance(providers, list)
    for row in providers:
        if isinstance(row, dict) and row.get("provider_id") == provider_id:
            return row
    raise AssertionError(f"Provider row not found: {provider_id}")


def test_online_governance_summary_marks_configured_remote_paths_inactive_in_offline_demo(tmp_path: Path) -> None:
    """Configured remote flags should stay visible even when offline_demo keeps them inactive."""

    config = _make_config(
        tmp_path,
        project_name="universal-agentic-data-pipeline-fitness-demo",
        source=SourceConfig(use_huggingface=True, use_github_search=True),
    )

    summary = build_online_governance_summary(config, [])

    assert summary["remote_sources_enabled"] is False
    assert summary["providers_requiring_attention"] == []
    assert "inactive" in summary["notes"][0]
    assert _provider(summary, "huggingface_datasets")["observed_status"] == "configured_but_inactive_for_runtime"
    assert _provider(summary, "github_repository_search")["observed_status"] == "configured_but_inactive_for_runtime"


def test_online_governance_summary_flags_unauthenticated_github_in_online_mode(monkeypatch, tmp_path: Path) -> None:
    """Online GitHub discovery should surface unauthenticated rate-limit risk when no token exists."""

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    config = _make_config(
        tmp_path,
        source=SourceConfig(use_github_search=True),
        runtime=RuntimeConfig(mode="online"),
    )

    summary = build_online_governance_summary(config, [])
    github = _provider(summary, "github_repository_search")

    assert summary["github_auth_mode"] == "unauthenticated"
    assert "github_repository_search" in summary["providers_requiring_attention"]
    assert github["active_in_runtime"] is True
    assert github["observed_status"] == "active_no_candidates"
    assert "GITHUB_TOKEN" in github["operator_action"]
    assert "GITHUB_TOKEN" in summary["notes"][-1]


def test_online_governance_summary_counts_candidates_by_provider(tmp_path: Path) -> None:
    """Candidate counts should be attributed to the correct provider buckets."""

    config = _make_config(
        tmp_path,
        source=SourceConfig(
            use_huggingface=True,
            use_github_search=True,
            use_internal_api=True,
            use_public_api=True,
            use_scraping_fallback=True,
        ),
        runtime=RuntimeConfig(mode="hybrid"),
    )

    candidates = [
        SourceCandidate("hf-1", "hf_dataset", "HF", "owner/name", metadata={"web_url": "https://huggingface.co/datasets/owner/name"}),
        SourceCandidate("gh-1", "github_repo", "GitHub", "https://github.com/octocat/repo"),
        SourceCandidate("api-int", "api", "Internal", "https://example.internal/api", metadata={"api_kind": "internal"}),
        SourceCandidate("api-pub", "api", "Public", "https://example.com/api", metadata={"api_kind": "public"}),
        SourceCandidate("scrape-web", "scrape", "Web", "https://example.com/reviews", metadata={"source_kind": "web_stub"}),
        SourceCandidate("scrape-demo", "scrape", "Demo", "demo://fitness", metadata={"demo_mode": True}),
    ]

    summary = build_online_governance_summary(config, candidates)

    assert _provider(summary, "huggingface_datasets")["discovered_candidates"] == 1
    assert _provider(summary, "github_repository_search")["discovered_candidates"] == 1
    assert _provider(summary, "internal_api_candidate")["discovered_candidates"] == 1
    assert _provider(summary, "public_api_candidate")["discovered_candidates"] == 1
    assert _provider(summary, "scraping_fallback_candidate")["discovered_candidates"] == 1
