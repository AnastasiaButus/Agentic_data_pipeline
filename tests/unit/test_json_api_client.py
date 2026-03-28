"""Unit tests for the generic JSON API client used by collection."""

from __future__ import annotations

import pytest

from src.providers.apis.json_api_client import JsonAPIClient


class FakeResponse:
    """Small response double that mimics the requests interface used by the client."""

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.raise_called = False

    def raise_for_status(self) -> None:
        self.raise_called = True

    def json(self) -> object:
        return self.payload


def test_fetch_json_uses_get_request_and_returns_payload(monkeypatch) -> None:
    """GET-based fetch_json should forward params and decode the JSON body."""

    captured: dict[str, object] = {}

    def fake_request(method: str, endpoint: str, **kwargs: object) -> FakeResponse:
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["kwargs"] = kwargs
        return FakeResponse({"items": [{"text": "A"}]})

    monkeypatch.setattr("src.providers.apis.json_api_client.requests.request", fake_request)

    client = JsonAPIClient(timeout=11)
    payload = client.fetch_json(
        "https://example.com/api/reviews",
        params={"topic": "fitness"},
        headers={"Authorization": "Bearer token"},
    )

    assert payload == {"items": [{"text": "A"}]}
    assert captured["method"] == "GET"
    assert captured["endpoint"] == "https://example.com/api/reviews"
    assert captured["kwargs"] == {
        "params": {"topic": "fitness"},
        "headers": {"Authorization": "Bearer token"},
        "json": None,
        "data": None,
        "timeout": 11,
    }


def test_request_rejects_empty_endpoint() -> None:
    """Empty endpoints should fail fast instead of issuing malformed HTTP requests."""

    client = JsonAPIClient()

    with pytest.raises(ValueError, match="endpoint must not be empty"):
        client.request("GET", "")
