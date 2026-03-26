"""GitHub REST API client used for repository and content lookups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    class _RequestsShim:
        """Fallback shim so tests can monkeypatch requests.get without the dependency."""

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("requests is required to call GitHub APIs")

    requests = _RequestsShim()


@dataclass(slots=True)
class GitHubClient:
    """Small HTTP client for GitHub REST endpoints."""

    token: str | None = None
    timeout: int = 20

    def search_repositories(self, query: str, per_page: int = 10) -> dict:
        """Search GitHub repositories using the REST API."""

        if not query:
            raise ValueError("query must not be empty")

        url = "https://api.github.com/search/repositories"
        response = requests.get(
            url,
            params={"q": query, "per_page": per_page},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._response_json(response)

    def get_repo_contents(self, owner: str, repo: str, path: str = "") -> dict:
        """Fetch repository contents from the GitHub contents API."""

        endpoint = f"https://api.github.com/repos/{owner}/{repo}/contents"
        cleaned_path = path.strip("/")
        if cleaned_path:
            endpoint = f"{endpoint}/{cleaned_path}"

        response = requests.get(
            endpoint,
            headers=self._contents_headers(),
            timeout=self.timeout,
        )
        return self._response_json(response)

    def _headers(self) -> dict[str, str]:
        """Build request headers and include auth only when a token is available."""

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "universal-agentic-data-pipeline",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _contents_headers(self) -> dict[str, str]:
        """Build headers that request GitHub contents in a consistent object format."""

        headers = self._headers()
        headers["Accept"] = "application/vnd.github.object+json"
        return headers

    def _response_json(self, response: Any) -> dict:
        """Raise on HTTP errors and return the decoded JSON payload."""

        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("GitHub API responses must decode to a JSON object")
        return payload
