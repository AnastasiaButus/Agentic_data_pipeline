"""Generic JSON API client for public or internal collection endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    class _RequestsShim:
        """Fallback shim so tests can monkeypatch requests.request without the dependency."""

        def request(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("requests is required to call generic JSON APIs")

    requests = _RequestsShim()


@dataclass(slots=True)
class JsonAPIClient:
    """Small HTTP client for JSON-based public or internal APIs."""

    timeout: int = 20

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Send one HTTP request and return the raw response object."""

        cleaned_endpoint = str(endpoint or "").strip()
        if not cleaned_endpoint:
            raise ValueError("endpoint must not be empty")

        response = requests.request(
            str(method or "GET").strip().upper() or "GET",
            cleaned_endpoint,
            params=dict(params or {}),
            headers=dict(headers or {}),
            json=json,
            data=data,
            timeout=timeout if timeout is not None else self.timeout,
        )
        response.raise_for_status()
        return response

    def fetch_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Fetch JSON from an endpoint using a GET request."""

        response = self.request(
            "GET",
            endpoint,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        return response.json()
