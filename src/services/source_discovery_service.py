"""Discover and rank candidate sources for the pipeline demo domain."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from src.core.context import PipelineContext
from src.core.runtime import (
    demo_key_for_config,
    runtime_allows_demo_sources,
    runtime_allows_remote_sources,
)
from src.domain import SourceCandidate
from src.services.artifact_registry import ArtifactRegistry
from src.services.source_compliance import (
    build_candidate_compliance_metadata,
    extract_github_license,
    extract_huggingface_license,
)
from src.services.source_governance import build_online_governance_summary


class SourceDiscoveryService:
    """Produce ranked source candidates using offline demo and narrow online discovery."""

    def __init__(self, ctx: PipelineContext, github_client: Any | None = None, registry: ArtifactRegistry | None = None) -> None:
        """Bind the discovery service to the active execution context."""

        self.ctx = ctx
        self.github_client = github_client
        self.registry = registry if registry is not None else ArtifactRegistry(ctx)
        self._last_online_governance_summary: dict[str, Any] = build_online_governance_summary(ctx.config, [])

    def search_huggingface(self) -> list[SourceCandidate]:
        """Return Hugging Face dataset candidates, preferring the real search path when possible."""

        source_config = getattr(getattr(self.ctx, "config", None), "source", None)
        if not bool(getattr(source_config, "use_huggingface", False)):
            return []
        if not runtime_allows_remote_sources(self.ctx.config):
            return []

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
            web_url = f"https://huggingface.co/datasets/{dataset_id}"
            metadata.update(
                build_candidate_compliance_metadata(
                    "hf_dataset",
                    web_url,
                    metadata=metadata,
                    license_label=extract_huggingface_license(item),
                )
            )

            candidates.append(
                SourceCandidate(
                    source_id=dataset_id,
                    source_type="hf_dataset",
                    title=title,
                    uri=dataset_id,
                    score=score,
                    metadata={**metadata, "web_url": web_url},
                )
            )

        return candidates

    def search_internal_apis(self) -> list[SourceCandidate]:
        """Return deterministic candidates for hidden or internal APIs."""

        source_config = getattr(getattr(self.ctx, "config", None), "source", None)
        if not bool(getattr(source_config, "use_internal_api", False)):
            return []
        if self._demo_key() is None:
            return []

        return [
            SourceCandidate(
                source_id="internal_reviews_api",
                source_type="api",
                title="Internal Reviews API",
                uri="https://example.internal/api/reviews",
                score=0.93,
                metadata=build_candidate_compliance_metadata(
                    "api",
                    "https://example.internal/api/reviews",
                    metadata={"api_kind": "internal"},
                )
                | {"api_kind": "internal"},
            )
        ]

    def search_public_apis(self) -> list[SourceCandidate]:
        """Return deterministic candidates for public APIs."""

        source_config = getattr(getattr(self.ctx, "config", None), "source", None)
        if not bool(getattr(source_config, "use_public_api", False)):
            return []
        if self._demo_key() is None:
            return []

        return [
            SourceCandidate(
                source_id="public_reviews_api",
                source_type="api",
                title="Public Reviews API",
                uri="https://example.com/api/reviews",
                score=0.9,
                metadata=build_candidate_compliance_metadata(
                    "api",
                    "https://example.com/api/reviews",
                    metadata={"api_kind": "public"},
                )
                | {"api_kind": "public"},
            )
        ]

    def search_github_repos(self) -> list[SourceCandidate]:
        """Search GitHub repositories for the current topic and map results to candidates."""

        source_config = getattr(getattr(self.ctx, "config", None), "source", None)
        if not bool(getattr(source_config, "use_github_search", False)):
            return []
        if not runtime_allows_remote_sources(self.ctx.config):
            return []
        topic = str(getattr(self.ctx.config.request, "topic", "")).strip()
        if not topic:
            return []

        try:
            if self.github_client is not None:
                response = self.github_client.search_repositories(topic, per_page=10)
            else:
                response = self._fetch_github_repositories(topic)
        except Exception:
            return []

        if not isinstance(response, dict):
            return []

        items = response.get("items")
        if not isinstance(items, list):
            return []

        candidates: list[SourceCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            source_id = item.get("full_name") or item.get("name") or item.get("html_url") or "github_repo"
            title = item.get("full_name") or item.get("name") or source_id
            uri = item.get("html_url") or item.get("clone_url") or item.get("url") or ""
            score = self._score_github_candidate(item.get("stargazers_count"), item.get("score"))
            metadata = {"source_kind": "github_search"}

            stars = item.get("stargazers_count")
            language = item.get("language")
            description = item.get("description")
            topics = item.get("topics") if isinstance(item.get("topics"), list) else []

            if stars is not None:
                metadata["stars"] = stars
            if language:
                metadata["language"] = language
            if description:
                metadata["description"] = description
            if topics:
                metadata["topics"] = topics[:5]
            metadata.update(
                build_candidate_compliance_metadata(
                    "github_repo",
                    uri,
                    metadata=metadata,
                    license_label=extract_github_license(item),
                )
            )

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

        source_config = getattr(getattr(self.ctx, "config", None), "source", None)
        if not bool(getattr(source_config, "use_scraping_fallback", False)):
            return []
        if self._demo_key() is None:
            return []

        return [
            SourceCandidate(
                source_id="web_reviews_stub",
                source_type="scrape",
                title="Fitness Supplements Review Page",
                uri="https://example.com/reviews",
                score=0.2,
                metadata=build_candidate_compliance_metadata(
                    "scrape",
                    "https://example.com/reviews",
                    metadata={"source_kind": "web_stub"},
                )
                | {"source_kind": "web_stub"},
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

        candidates: list[SourceCandidate] = []
        if runtime_allows_demo_sources(self.ctx.config):
            candidates.extend(self._demo_candidates())

        if runtime_allows_remote_sources(self.ctx.config):
            candidates.extend(
                self.search_huggingface()
                + self.search_internal_apis()
                + self.search_public_apis()
                + self.search_github_repos()
                + self.search_web_pages_for_scraping()
            )

        ranked = self.rank_candidates(candidates)
        ranked = self._limit_candidates(ranked)
        self._last_online_governance_summary = build_online_governance_summary(self.ctx.config, ranked)
        self.registry.save_json("data/raw/discovered_sources.json", [candidate.as_dict() for candidate in ranked])
        return ranked

    def get_online_governance_summary(self, candidates: list[SourceCandidate] | None = None) -> dict[str, Any]:
        """Return the last computed online governance summary or rebuild it from candidates."""

        if candidates is None:
            return dict(self._last_online_governance_summary)
        self._last_online_governance_summary = build_online_governance_summary(self.ctx.config, candidates)
        return dict(self._last_online_governance_summary)

    def load_approved_source_ids(self, path: str | Path = "data/raw/approved_sources.json") -> list[str] | None:
        """Load approved source ids from a simple JSON list.

        The approval MVP uses a flat list of source_id strings so it stays easy to inspect and
        test. Missing approval files are treated as "no approval gate present".
        """

        if not self.registry.exists(path):
            return None

        payload = self.registry.load_json(path)
        if not isinstance(payload, list):
            return []

        approved_ids: list[str] = []
        for item in payload:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    approved_ids.append(normalized)

        return approved_ids

    def filter_approved_candidates(
        self,
        candidates: list[SourceCandidate],
        approved_source_ids: list[str] | None = None,
        path: str | Path = "data/raw/approved_sources.json",
    ) -> list[SourceCandidate]:
        """Filter a shortlist down to approved candidates when an approval file exists.

        If the approval file is missing, the shortlist is returned unchanged so the baseline flow
        remains predictable. Unknown approved ids are ignored safely.
        """

        ids = approved_source_ids if approved_source_ids is not None else self.load_approved_source_ids(path)
        if ids is None:
            return list(candidates)

        approved_set = {source_id for source_id in ids if source_id}
        if not approved_set:
            return []

        return [candidate for candidate in candidates if candidate.source_id in approved_set]

    def load_approved_candidates(
        self,
        candidates: list[SourceCandidate],
        path: str | Path = "data/raw/approved_sources.json",
    ) -> list[SourceCandidate]:
        """Return the approved subset for a shortlist using the approval file MVP."""

        return self.filter_approved_candidates(candidates, path=path)

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
                    metadata=build_candidate_compliance_metadata(
                        "scrape",
                        "demo://fitness-supplements",
                        metadata={"html": self._fitness_demo_html(), "demo_mode": True, "topic": self.ctx.config.request.topic},
                    )
                    | {"html": self._fitness_demo_html(), "demo_mode": True, "topic": self.ctx.config.request.topic},
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
                    metadata=build_candidate_compliance_metadata(
                        "scrape",
                        "demo://minecraft-instructions",
                        metadata={"html": self._minecraft_demo_html(), "demo_mode": True, "topic": self.ctx.config.request.topic},
                    )
                    | {"html": self._minecraft_demo_html(), "demo_mode": True, "topic": self.ctx.config.request.topic},
                )
            ]

        return []

    def _demo_key(self) -> str | None:
        """Map the active config to a known offline demo payload when appropriate."""

        return demo_key_for_config(self.ctx.config)

    def _limit_candidates(self, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        """Apply the configured shortlist cap after ranking."""

        source_config = getattr(getattr(self.ctx, "config", None), "source", None)
        max_sources = int(getattr(source_config, "max_sources", 0) or 0)
        if max_sources <= 0:
            return list(candidates)
        return list(candidates[:max_sources])

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

    def _fetch_github_repositories(self, topic: str) -> dict[str, Any]:
        """Fetch the GitHub repository search payload for a topic.

        The helper is isolated so tests can monkeypatch it and production discovery stays narrow.
        """

        url = f"https://api.github.com/search/repositories?q={quote_plus(topic)}&sort=stars&order=desc&per_page=10"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "universal-agentic-data-pipeline",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        github_token = os.getenv("GITHUB_TOKEN", "").strip()
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        request = Request(
            url,
            headers=headers,
        )
        with urlopen(request, timeout=5) as response:
            raw = response.read().decode("utf-8")
        payload = json.loads(raw)
        if isinstance(payload, list):
            return {"items": payload}
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

    def _score_github_candidate(self, stargazers_count: Any, search_score: Any) -> float:
        """Convert GitHub popularity signals into a compact ranking score."""

        return max(self._coerce_float(stargazers_count), self._coerce_float(search_score))

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
            '<div class="review" data-text="Minecraft guide: crafting instructions for redstone tools and starter builds" data-rating="5">Helpful crafting guide</div>'
            '<div class="review" data-text="Minecraft guide: combat warning about potion timing in arena fights" data-rating="2">Combat warning note</div>'
            '<div class="review" data-text="Minecraft guide: enchantments improve armor and tool progression" data-rating="4">Enchantments overview</div>'
            '</body></html>'
        )
