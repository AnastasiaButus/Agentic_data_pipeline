"""Discover and rank candidate sources for the pipeline demo domain."""

from __future__ import annotations

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
        """Return a deterministic Hugging Face dataset candidate."""

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

    def search_internal_apis(self) -> list[SourceCandidate]:
        """Return deterministic candidates for hidden or internal APIs."""

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
            return [
                SourceCandidate(
                    source_id="github_reviews_stub",
                    source_type="github_repo",
                    title="Fitness Reviews Repo",
                    uri="https://github.com/example/fitness-reviews",
                    score=0.7,
                    metadata={"source_kind": "github_stub"},
                )
            ]

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
