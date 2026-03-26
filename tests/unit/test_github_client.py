"""Tests for the GitHub REST API client provider."""

from __future__ import annotations

import pytest

from src.providers.apis.github_client import GitHubClient


class _FakeResponse:
    """Simple response double used to capture GitHub API requests."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.raise_called = False

    def raise_for_status(self) -> None:
        self.raise_called = True

    def json(self) -> dict:
        return self.payload


def test_github_search_repositories_formulates_request(monkeypatch) -> None:
    """The search endpoint should use the expected URL, params, and headers."""

    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse({"items": []})

    monkeypatch.setattr("src.providers.apis.github_client.requests.get", fake_get)

    client = GitHubClient(token="secret-token", timeout=12)
    payload = client.search_repositories("fitness supplements", per_page=7)

    assert payload == {"items": []}
    assert captured["url"] == "https://api.github.com/search/repositories"
    assert captured["kwargs"]["params"] == {"q": "fitness supplements", "per_page": 7}
    assert captured["kwargs"]["timeout"] == 12
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer secret-token"


def test_github_repo_contents_formulates_endpoint(monkeypatch) -> None:
    """The contents endpoint should be built with the owner, repo, and optional path."""

    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse({"name": "README.md"})

    monkeypatch.setattr("src.providers.apis.github_client.requests.get", fake_get)

    client = GitHubClient()
    payload = client.get_repo_contents("octocat", "hello-world", "docs/readme.md")

    assert payload == {"name": "README.md"}
    assert captured["url"] == "https://api.github.com/repos/octocat/hello-world/contents/docs/readme.md"
    assert captured["kwargs"]["headers"]["Accept"] == "application/vnd.github.object+json"


def test_github_repo_contents_handles_directory_objects(monkeypatch) -> None:
    """Directory contents should still decode to a consistent object response."""

    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse({"entries": [{"name": "README.md"}], "type": "dir"})

    monkeypatch.setattr("src.providers.apis.github_client.requests.get", fake_get)

    client = GitHubClient()
    payload = client.get_repo_contents("octocat", "hello-world", "docs")

    assert payload == {"entries": [{"name": "README.md"}], "type": "dir"}
    assert captured["url"] == "https://api.github.com/repos/octocat/hello-world/contents/docs"
    assert captured["kwargs"]["headers"]["Accept"] == "application/vnd.github.object+json"


def test_github_repo_contents_without_token(monkeypatch) -> None:
    """The client should work without auth headers when no token is configured."""

    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        captured["kwargs"] = kwargs
        return _FakeResponse({"name": "README.md"})

    monkeypatch.setattr("src.providers.apis.github_client.requests.get", fake_get)

    client = GitHubClient(token=None)
    payload = client.get_repo_contents("octocat", "hello-world")

    assert payload == {"name": "README.md"}
    assert "Authorization" not in captured["kwargs"]["headers"]


def test_github_search_empty_query_raises_value_error() -> None:
    """An empty search query should fail fast before any HTTP request is made."""

    client = GitHubClient()

    with pytest.raises(ValueError):
        client.search_repositories("")
