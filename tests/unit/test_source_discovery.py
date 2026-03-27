"""Tests for source discovery and ranking behavior."""

from __future__ import annotations

from pathlib import Path

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.domain import SourceCandidate
from src.services.source_discovery_service import SourceDiscoveryService


class FakeRegistry:
    """Capture persisted discovery payloads without touching the filesystem."""

    def __init__(self) -> None:
        self.saved: tuple[str, object] | None = None
        self.files: dict[str, object] = {}

    def save_json(self, path: str | Path, payload: object) -> Path:
        normalized_path = str(path)
        self.saved = (normalized_path, payload)
        self.files[normalized_path] = payload
        return Path(path)

    def exists(self, path: str | Path) -> bool:
        return str(path) in self.files

    def load_json(self, path: str | Path) -> object:
        return self.files[str(path)]


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for discovery tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def test_rank_candidates_sorts_by_expected_priority() -> None:
    """API candidates should outrank datasets, GitHub repos, and scraping fallback."""

    service = SourceDiscoveryService(_make_context(Path(".")))
    ranked = service.rank_candidates(
        [
            SourceCandidate("scrape-1", "scrape", "Scrape", "https://example.com", score=0.99),
            SourceCandidate("github-1", "github_repo", "GitHub", "https://github.com/a", score=0.5),
            SourceCandidate("hf-1", "hf_dataset", "HF", "dataset/name", score=0.7),
            SourceCandidate("api-low", "api", "API low", "https://api/a", score=0.2),
            SourceCandidate("api-high", "api", "API high", "https://api/b", score=0.9),
        ]
    )

    assert [candidate.source_id for candidate in ranked] == ["api-high", "api-low", "hf-1", "github-1", "scrape-1"]


def test_rank_candidates_empty_list_returns_empty_list() -> None:
    """Ranking an empty candidate list should remain empty."""

    service = SourceDiscoveryService(_make_context(Path(".")))

    assert service.rank_candidates([]) == []


def test_run_saves_discovered_sources_json(monkeypatch, tmp_path: Path) -> None:
    """run() should persist the ranked discovery payload as discovered_sources.json."""

    context = _make_context(tmp_path)
    registry = FakeRegistry()
    service = SourceDiscoveryService(context, registry=registry)

    monkeypatch.setattr(service, "search_huggingface", lambda: [SourceCandidate("hf-1", "hf_dataset", "HF", "dataset/name", score=0.5)])
    monkeypatch.setattr(service, "search_internal_apis", lambda: [SourceCandidate("api-1", "api", "API", "https://api/a", score=0.8, metadata={"api_kind": "internal"})])
    monkeypatch.setattr(service, "search_public_apis", lambda: [])
    monkeypatch.setattr(service, "search_web_pages_for_scraping", lambda: [])
    monkeypatch.setattr(service, "search_github_repos", lambda: [])

    ranked = service.run()

    assert [candidate.source_id for candidate in ranked] == ["api-1", "hf-1"]
    assert registry.saved is not None
    assert registry.saved[0] == "data/raw/discovered_sources.json"
    payload = registry.saved[1]
    assert isinstance(payload, list)
    assert payload[0]["source_id"] == "api-1"


def test_search_github_repos_transforms_response_to_candidates(monkeypatch, tmp_path: Path) -> None:
    """GitHub search results should become github_repo SourceCandidate objects."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    service = SourceDiscoveryService(context)

    captured: dict[str, object] = {}
    payload = {
        "items": [
            {
                "full_name": "octocat/fitness-reviews",
                "html_url": "https://github.com/octocat/fitness-reviews",
                "stargazers_count": 17,
                "language": "Python",
                "description": "Repository for review tooling",
                "topics": ["reviews", "fitness", "pipeline"],
                "directory": "docs",
                "entries": [{"name": "README.md"}],
            }
        ]
    }

    def fake_fetch(topic: str) -> dict:
        captured["topic"] = topic
        return payload

    monkeypatch.setattr(service, "_fetch_github_repositories", fake_fetch)
    candidates = service.search_github_repos()

    assert len(candidates) == 1
    assert candidates[0].source_type == "github_repo"
    assert candidates[0].source_id == "octocat/fitness-reviews"
    assert candidates[0].title == "octocat/fitness-reviews"
    assert candidates[0].uri == "https://github.com/octocat/fitness-reviews"
    assert candidates[0].score == 17.0
    assert candidates[0].metadata == {
        "source_kind": "github_search",
        "stars": 17,
        "language": "Python",
        "description": "Repository for review tooling",
        "topics": ["reviews", "fitness", "pipeline"],
    }
    assert captured["topic"] == "fitness supplements"


def test_search_github_repos_uses_topic_from_request_config(monkeypatch, tmp_path: Path) -> None:
    """GitHub search should use the topic from the request config rather than a hardcoded string."""

    captured: dict[str, object] = {}

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements reviews"

    service = SourceDiscoveryService(context)

    def fake_fetch(topic: str) -> dict:
        captured["query"] = topic
        return {"items": []}

    monkeypatch.setattr(service, "_fetch_github_repositories", fake_fetch)
    service.search_github_repos()

    assert captured["query"] == "fitness supplements reviews"


def test_non_demo_shortlist_omits_fake_stub_candidates(monkeypatch, tmp_path: Path) -> None:
    """Non-demo discovery should not auto-add fake API, GitHub, or scrape stubs."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    registry = FakeRegistry()
    service = SourceDiscoveryService(context, registry=registry)

    monkeypatch.setattr(service, "_fetch_huggingface_datasets", lambda topic: {
        "datasets": [
            {
                "id": "fitness/supplements-reviews",
                "title": "Fitness Supplements Reviews",
                "downloads": 1234,
                "likes": 56,
                "tags": ["text-classification", "reviews"],
            }
        ]
    })
    monkeypatch.setattr(service, "_fetch_github_repositories", lambda topic: {"items": []})

    ranked = service.run()

    assert [candidate.source_type for candidate in ranked] == ["hf_dataset"]
    assert registry.saved is not None
    payload = registry.saved[1]
    assert isinstance(payload, list)
    assert [row["source_type"] for row in payload] == ["hf_dataset"]


def test_search_github_repos_failure_falls_back_safely(monkeypatch, tmp_path: Path) -> None:
    """GitHub discovery should return an empty list when lookup fails."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    service = SourceDiscoveryService(context)

    monkeypatch.setattr(service, "_fetch_github_repositories", lambda topic: (_ for _ in ()).throw(RuntimeError("network down")))

    assert service.search_github_repos() == []


def test_search_github_repos_invalid_items_payload_returns_empty_list(monkeypatch, tmp_path: Path) -> None:
    """Malformed GitHub payloads should return an empty shortlist without raising."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    service = SourceDiscoveryService(context)

    payloads = [
        {},
        {"items": None},
        {"items": "not-a-list"},
    ]

    for payload in payloads:
        monkeypatch.setattr(service, "_fetch_github_repositories", lambda topic, p=payload: p)
        assert service.search_github_repos() == []


def test_search_github_repos_rate_limit_like_payload_returns_empty_list(monkeypatch, tmp_path: Path) -> None:
    """GitHub error-like payloads should return an empty shortlist without raising."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    service = SourceDiscoveryService(context)

    payloads = [
        {"message": "API rate limit exceeded for 1.2.3.4"},
        {
            "message": "Requires authentication",
            "documentation_url": "https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting",
        },
    ]

    for payload in payloads:
        monkeypatch.setattr(service, "_fetch_github_repositories", lambda topic, p=payload: p)
        assert service.search_github_repos() == []


def test_internal_api_candidates_use_api_source_type(tmp_path: Path) -> None:
    """Internal API discovery should keep the generic api source type and mark kind in metadata."""

    config = AppConfig(
        project=ProjectConfig(name="universal-agentic-data-pipeline-fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    service = SourceDiscoveryService(PipelineContext.from_config(config))
    candidates = service.search_internal_apis()

    assert candidates[0].source_type == "api"
    assert candidates[0].metadata["api_kind"] == "internal"


def test_non_demo_stub_methods_return_empty_lists(tmp_path: Path) -> None:
    """Non-demo configs should not get fake API or scrape stub candidates."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    service = SourceDiscoveryService(context)

    assert service.search_internal_apis() == []
    assert service.search_public_apis() == []
    assert service.search_web_pages_for_scraping() == []


def test_approved_sources_file_filters_shortlist_by_source_id(monkeypatch, tmp_path: Path) -> None:
    """Approved source ids should filter the discovered shortlist deterministically."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    registry = FakeRegistry()
    service = SourceDiscoveryService(context, registry=registry)

    monkeypatch.setattr(service, "_fetch_huggingface_datasets", lambda topic: {
        "datasets": [
            {
                "id": "fitness/supplements-reviews",
                "title": "Fitness Supplements Reviews",
                "downloads": 1234,
                "likes": 56,
                "tags": ["text-classification", "reviews"],
            },
            {
                "id": "fitness/other-dataset",
                "title": "Other Dataset",
                "downloads": 222,
                "likes": 1,
                "tags": ["reviews"],
            },
        ]
    })

    discovered = service.run()
    registry.save_json("data/raw/approved_sources.json", ["fitness/supplements-reviews"])

    approved = service.filter_approved_candidates(discovered)

    assert [candidate.source_id for candidate in approved] == ["fitness/supplements-reviews"]


def test_unknown_approved_source_ids_are_ignored_safely(monkeypatch, tmp_path: Path) -> None:
    """Unknown approved ids should be ignored without breaking the approval helper."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    registry = FakeRegistry()
    service = SourceDiscoveryService(context, registry=registry)

    monkeypatch.setattr(service, "_fetch_huggingface_datasets", lambda topic: {
        "datasets": [
            {"id": "fitness/supplements-reviews", "title": "Fitness Supplements Reviews", "downloads": 1234, "likes": 56, "tags": ["text-classification", "reviews"]},
            {"id": "fitness/other-dataset", "title": "Other Dataset", "downloads": 222, "likes": 1, "tags": ["reviews"]},
        ]
    })

    discovered = service.run()
    registry.save_json("data/raw/approved_sources.json", ["missing-id", "fitness/other-dataset"])

    approved = service.filter_approved_candidates(discovered)

    assert [candidate.source_id for candidate in approved] == ["fitness/other-dataset"]


def test_missing_approved_sources_file_returns_original_shortlist(monkeypatch, tmp_path: Path) -> None:
    """Absence of approved_sources.json should not alter the shortlist helper output."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    service = SourceDiscoveryService(context)

    shortlist = [
        SourceCandidate("hf-1", "hf_dataset", "HF 1", "dataset/one", score=1.0),
        SourceCandidate("api-1", "api", "API 1", "https://api/one", score=0.5),
    ]

    assert service.filter_approved_candidates(shortlist) == shortlist


def test_demo_config_keeps_offline_demo_path(tmp_path: Path) -> None:
    """Explicit demo configs should continue to use the offline demo discovery path."""

    config = AppConfig(
        project=ProjectConfig(name="universal-agentic-data-pipeline-fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    context = PipelineContext.from_config(config)
    service = SourceDiscoveryService(context)

    discovered = service.run()

    assert discovered[0].source_type == "scrape"
    assert discovered[0].uri == "demo://fitness-supplements"


def test_non_demo_config_uses_real_huggingface_path(monkeypatch, tmp_path: Path) -> None:
    """Non-demo configs should call the real Hugging Face search helper."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    service = SourceDiscoveryService(context)

    payload = {
        "datasets": [
            {
                "id": "fitness/supplements-reviews",
                "title": "Fitness Supplements Reviews",
                "downloads": 1234,
                "likes": 56,
                "tags": ["text-classification", "reviews"],
            }
        ]
    }
    monkeypatch.setattr(service, "_fetch_huggingface_datasets", lambda topic: payload)

    candidates = service.search_huggingface()

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source_type == "hf_dataset"
    assert candidate.source_id == "fitness/supplements-reviews"
    assert candidate.title == "Fitness Supplements Reviews"
    assert candidate.uri == "fitness/supplements-reviews"
    assert candidate.metadata["source_kind"] == "hf_search"
    assert candidate.metadata["web_url"] == "https://huggingface.co/datasets/fitness/supplements-reviews"
    assert candidate.metadata["downloads"] == 1234
    assert candidate.metadata["likes"] == 56
    assert candidate.metadata["tags"] == ["text-classification", "reviews"]


def test_non_demo_shortlist_can_include_hf_and_github_real_candidates(monkeypatch, tmp_path: Path) -> None:
    """Non-demo discovery should be able to shortlist both HF and GitHub candidates."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    registry = FakeRegistry()
    service = SourceDiscoveryService(context, registry=registry)

    monkeypatch.setattr(service, "_fetch_huggingface_datasets", lambda topic: {
        "datasets": [
            {
                "id": "fitness/supplements-reviews",
                "title": "Fitness Supplements Reviews",
                "downloads": 1234,
                "likes": 56,
                "tags": ["text-classification", "reviews"],
            }
        ]
    })
    monkeypatch.setattr(service, "_fetch_github_repositories", lambda topic: {
        "items": [
            {
                "full_name": "octocat/fitness-reviews",
                "html_url": "https://github.com/octocat/fitness-reviews",
                "stargazers_count": 17,
                "language": "Python",
                "description": "Repository for review tooling",
            }
        ]
    })

    ranked = service.run()

    assert [candidate.source_type for candidate in ranked] == ["hf_dataset", "github_repo"]
    assert [candidate.source_id for candidate in ranked] == ["fitness/supplements-reviews", "octocat/fitness-reviews"]
    assert registry.saved is not None
    payload = registry.saved[1]
    assert isinstance(payload, list)
    assert [row["source_type"] for row in payload] == ["hf_dataset", "github_repo"]


def test_real_huggingface_candidate_uses_canonical_dataset_id_uri(monkeypatch, tmp_path: Path) -> None:
    """Real HF discovery should store canonical dataset ids in uri and the page URL in metadata."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    service = SourceDiscoveryService(context)

    payload = {
        "datasets": [
            {
                "id": "owner/name",
                "title": "Owner Name Dataset",
                "downloads": 10,
                "likes": 1,
                "tags": ["text-classification"],
            }
        ]
    }
    monkeypatch.setattr(service, "_fetch_huggingface_datasets", lambda topic: payload)

    candidate = service.search_huggingface()[0]

    assert candidate.source_id == "owner/name"
    assert candidate.uri == "owner/name"
    assert candidate.metadata["web_url"] == "https://huggingface.co/datasets/owner/name"


def test_real_huggingface_path_failure_falls_back_safely(monkeypatch, tmp_path: Path) -> None:
    """Network failures in the real Hugging Face path should not break run()."""

    context = _make_context(tmp_path)
    context.config.project.name = "non-demo"
    context.config.request.topic = "fitness supplements"
    registry = FakeRegistry()
    service = SourceDiscoveryService(context, registry=registry)

    monkeypatch.setattr(service, "_fetch_huggingface_datasets", lambda topic: (_ for _ in ()).throw(RuntimeError("network down")))
    monkeypatch.setattr(service, "_fetch_github_repositories", lambda topic: (_ for _ in ()).throw(RuntimeError("network down")))

    ranked = service.run()

    assert ranked == []
    assert registry.saved is not None
    assert registry.saved[0] == "data/raw/discovered_sources.json"
    assert registry.saved[1] == []
