"""Protocols that describe the domain-facing integration points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .dataclasses import SourceCandidate


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Describe the minimal generation contract for an external LLM client."""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text response for the given prompt."""


@runtime_checkable
class DatasetLoaderProtocol(Protocol):
    """Describe a loader that can materialize datasets from a source candidate."""

    def load(self, candidate: SourceCandidate) -> Any:
        """Load one dataset or batch from a source candidate."""

    def load_many(self, candidates: Sequence[SourceCandidate]) -> list[Any]:
        """Load multiple candidates in a deterministic order."""


@runtime_checkable
class APIClientProtocol(Protocol):
    """Describe a generic HTTP client suitable for public or hidden API access."""

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
        """Send one HTTP request to a public or internal endpoint."""

    def fetch_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Fetch JSON from an endpoint with minimal caller assumptions."""


@runtime_checkable
class ArtifactRegistryProtocol(Protocol):
    """Describe a registry for storing and retrieving pipeline artifacts."""

    def save_artifact(self, name: str, payload: Any) -> str:
        """Persist an artifact and return its resolved reference."""

    def load_artifact(self, name: str) -> Any:
        """Load a previously stored artifact by name."""

    def list_artifacts(self, prefix: str | None = None) -> list[str]:
        """List registered artifacts, optionally constrained by prefix."""
