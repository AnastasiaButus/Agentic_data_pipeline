"""Discover and rank candidate sources for the pipeline demo domain."""

from __future__ import annotations

import json
from urllib.parse import quote_plus
from urllib.request import urlopen
from pathlib import Path
from typing import Any

from src.core.context import PipelineContext
from src.domain import SourceCandidate
from src.services.artifact_registry import ArtifactRegistry


class SourceDiscoveryService:
    """Produce ranked source candidates using deterministic discovery stubs."""

    def __init__(self, ctx: PipelineContext, github_client: Any | None = None, registry: ArtifactRegistry | None = None) -> None:
        """Bind the discovery service to the active execution context."""

        self.ctx = ctx
        self.github_client = github_client
        self.registry = registry if registry is not None else ArtifactRegistry(ctx)

    def search_huggingface(self) -> list[SourceCandidate]:
        """Return Hugging Face dataset candidates, preferring the real search path when possible."""

        demo_key = self._demo_key()
        if demo_key == "fitness":
            return [
                SourceCandidate(
                    source_id="hf_fitness_supplements_reviews",
                    source_type="hf_dataset",
                    title="Fitness Supplements Reviews Dataset",
                    uri="fitness-supplements/reviews",
                    score=0.95,
                    metadata={"domain": "fitness_supplements", "source_kind": "dataset"},
                )
            ]

        return self.search_huggingface_real()

    def search_huggingface_real(self) -> list[SourceCandidate]:
        """Search the public Hugging Face datasets API for the current topic.

        This is a narrow discovery-only MVP. It converts the API response into SourceCandidate
        rows and returns an empty list if the request fails for any reason.
        """

        topic = str(getattr(self.ctx.config.request, "topic", "")).strip()
        if not topic:
            return []

        try:
            payload = self._fetch_huggingface_datasets(topic)
        except Exception:
            return []

        items = payload.get("datasets") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []

        candidates: list[SourceCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            dataset_id = str(item.get("id") or item.get("dataset_id") or item.get("full_name") or "hf_dataset")
            title = str(item.get("title") or item.get("id") or dataset_id)
            downloads = item.get("downloads")
            likes = item.get("likes")
            tags = item.get("tags") if isinstance(item.get("tags"), list) else []
            score = self._score_huggingface_candidate(downloads, likes)
            metadata = {"source_kind": "hf_search"}

            if downloads is not None:
                metadata["downloads"] = downloads
            if likes is not None:
                metadata["likes"] = likes
            if tags:
                metadata["tags"] = tags

            candidates.append(
                SourceCandidate(
                    source_id=dataset_id,
                    source_type="hf_dataset",
                    title=title,
                    uri=f"https://huggingface.co/datasets/{dataset_id}",
                    score=score,
                    metadata=metadata,
                )
            )

        return candidates

    def search_internal_apis(self) -> list[SourceCandidate]:
        """Return deterministic candidates for hidden or internal APIs."""

        if self._demo_key() is None:
            return []

        return [
            SourceCandidate(
                source_id="internal_reviews_api",
                source_type="api",
                title="Internal Reviews API",
                uri="https://example.internal/api/reviews",
                score=0.93,
                metadata={"api_kind": "internal"},
            )
        ]

    def search_public_apis(self) -> list[SourceCandidate]:
        """Return deterministic candidates for public APIs."""

        if self._demo_key() is None:
            return []

        return [
            SourceCandidate(
                source_id="public_reviews_api",
                source_type="api",
                title="Public Reviews API",
                uri="https://example.com/api/reviews",
                score=0.9,
                metadata={"api_kind": "public"},
            )
        ]

    def search_github_repos(self) -> list[SourceCandidate]:
        """Use the configured GitHub client when present and map results to candidates."""

        if self.github_client is None:
            return []

        topic = self.ctx.config.request.topic
        response = self.github_client.search_repositories(topic, per_page=10)
        items = response.get("items") if isinstance(response, dict) else None
        if items is None:
            items = [response]

        candidates: list[SourceCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            source_id = item.get("full_name") or item.get("name") or item.get("html_url") or "github_repo"
            title = item.get("full_name") or item.get("name") or source_id
            uri = item.get("html_url") or item.get("clone_url") or item.get("url") or ""
            score = float(item.get("stargazers_count") or item.get("score") or 0.0)
            metadata = {key: value for key, value in item.items() if key not in {"full_name", "name", "html_url", "clone_url", "url", "stargazers_count", "score"}}

            candidates.append(
                SourceCandidate(
                    source_id=str(source_id),
                    source_type="github_repo",
                    title=str(title),
                    uri=str(uri),
                    score=score,
                    metadata=metadata,
                )
            )

        return candidates

    def search_web_pages_for_scraping(self) -> list[SourceCandidate]:
        """Return a deterministic scraping fallback candidate without implementing scraping."""

        if self._demo_key() is None:
            return []

        return [
            SourceCandidate(
                source_id="web_reviews_stub",
                source_type="scrape",
                title="Fitness Supplements Review Page",
                uri="https://example.com/reviews",
                score=0.2,
                metadata={"source_kind": "web_stub"},
            )
        ]

    def rank_candidates(self, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        """Rank candidates by source priority and then by score descending."""

        if not candidates:
            return []

        priority = {"api": 0, "hf_dataset": 1, "github_repo": 2, "scrape": 3}
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda item: (priority.get(item[1].source_type, 99), -item[1].score, item[0]))
        return [candidate for _, candidate in indexed]

    def run(self) -> list[SourceCandidate]:
        """Run discovery, rank the results, and persist the serialized output."""

        demo_candidates = self._demo_candidates()
        if demo_candidates:
            self.registry.save_json("data/raw/discovered_sources.json", [candidate.as_dict() for candidate in demo_candidates])
            return demo_candidates

        candidates = (
            self.search_huggingface()
            + self.search_internal_apis()
            + self.search_public_apis()
            + self.search_github_repos()
            + self.search_web_pages_for_scraping()
        )
        ranked = self.rank_candidates(candidates)
        self.registry.save_json("data/raw/discovered_sources.json", [candidate.as_dict() for candidate in ranked])
        return ranked

    def _demo_candidates(self) -> list[SourceCandidate]:
        """Return local, offline source candidates for the persistent demo configs."""

        demo_key = self._demo_key()
        if demo_key is None:
            return []

        if demo_key == "fitness":
            return [
                SourceCandidate(
                    source_id="demo_fitness_scrape",
                    source_type="scrape",
                    title="Fitness Supplements Offline Demo",
                    uri="demo://fitness-supplements",
                    score=1.0,
                    metadata={"html": self._fitness_demo_html(), "demo_mode": True, "topic": self.ctx.config.request.topic},
                )
            ]

        if demo_key == "minecraft":
            return [
                SourceCandidate(
                    source_id="demo_minecraft_scrape",
                    source_type="scrape",
                    title="Minecraft Instructions Offline Demo",
                    uri="demo://minecraft-instructions",
                    score=1.0,
                    metadata={"html": self._minecraft_demo_html(), "demo_mode": True, "topic": self.ctx.config.request.topic},
                )
            ]

        return []

    def _demo_key(self) -> str | None:
        """Map the active config to a known offline demo payload when appropriate."""

        project = getattr(self.ctx, "config", None)
        project_name = str(getattr(getattr(project, "project", None), "name", ""))

        # Demo mode is now keyed only by explicit project identity so future real-run
        # configs that reuse the same topic cannot accidentally enter the offline path.
        if project_name == "universal-agentic-data-pipeline-fitness-demo":
            return "fitness"
        if project_name == "universal-agentic-data-pipeline-minecraft-demo":
            return "minecraft"
        return None

    def _fetch_huggingface_datasets(self, topic: str) -> dict[str, Any]:
        """Fetch the Hugging Face datasets search payload for a topic.

        The helper is isolated so tests can monkeypatch it and production discovery stays narrow.
        """

        url = f"https://huggingface.co/api/datasets?search={quote_plus(topic)}&limit=10"
        with urlopen(url, timeout=5) as response:
            raw = response.read().decode("utf-8")
        payload = json.loads(raw)
        if isinstance(payload, list):
            return {"datasets": payload}
        return payload

    def _score_huggingface_candidate(self, downloads: Any, likes: Any) -> float:
        """Convert Hugging Face popularity signals into a compact ranking score."""

        numeric_downloads = self._coerce_float(downloads)
        numeric_likes = self._coerce_float(likes)
        return numeric_downloads + (10.0 * numeric_likes)

    def _coerce_float(self, value: Any) -> float:
        """Convert a value into a non-negative float while tolerating missing fields."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric != numeric:
            return 0.0
        return max(0.0, numeric)

    def _fitness_demo_html(self) -> str:
        """Return a local HTML payload with fitness supplement review blocks."""

        return (
            "<html><body>"
            '<div class="review" data-text="Fitness review: energy boost from a supplement" data-rating="5">Great energy and workout support</div>'
            '<div class="review" data-text="Fitness review: side effect warning after protein powder" data-rating="1">Upset stomach side effect</div>'
            '<div class="review" data-text="Fitness review: balanced supplement routine" data-rating="3">Neutral supplement experience</div>'
            '</body></html>'
        )

    def _minecraft_demo_html(self) -> str:
        """Return a local HTML payload with minecraft instruction review blocks."""

        return (
            "<html><body>"
            '<div class="review" data-text="Minecraft review: energy tip for long survival builds" data-rating="5">Helpful building guide</div>'
            '<div class="review" data-text="Minecraft review: side effect warning from risky potion use" data-rating="1">Potion side effect note</div>'
            '<div class="review" data-text="Minecraft review: crafting instructions for redstone tools" data-rating="3">Crafting instructions summary</div>'
            '</body></html>'
        )
