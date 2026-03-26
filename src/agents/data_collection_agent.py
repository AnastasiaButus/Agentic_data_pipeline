"""Collect raw data from configured sources and normalize it into the canonical schema."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.constants import STANDARD_COLUMNS
from src.domain import SourceCandidate
from src.providers.datasets.hf_loader import HFDatasetLoader
from src.providers.web.scraper import parse_review_blocks
from src.services.artifact_registry import ArtifactRegistry
from src.services.dataset_filter_service import filter_fitness_reviews
from src.services.schema_normalization_service import SchemaNormalizationService


class DataCollectionAgent(BaseAgent):
    """Collect and normalize source data without introducing a new architecture layer."""

    def __init__(
        self,
        ctx: Any,
        hf_loader: HFDatasetLoader | None = None,
        normalizer: SchemaNormalizationService | None = None,
        scraper: Any | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        """Bind the agent to the pipeline context and its helpers."""

        super().__init__(ctx, registry if registry is not None else ArtifactRegistry(ctx))
        self.hf_loader = hf_loader if hf_loader is not None else HFDatasetLoader()
        self.normalizer = normalizer if normalizer is not None else SchemaNormalizationService()
        self.scraper = scraper if scraper is not None else parse_review_blocks

    def run(self, sources: list[SourceCandidate]) -> Any:
        """Collect, merge, persist raw data, and return the normalized canonical frame."""

        if not sources:
            empty_raw = self._empty_frame()
            self.registry.save_dataframe("data/raw/merged_raw.parquet", empty_raw)
            return self.normalizer.normalize_reviews([], source_name=self.name, source_type="collect")

        raw_frames: list[Any] = []
        normalized_frames: list[Any] = []
        for source in sources:
            raw_frame = self._collect_source(source)
            if self._is_empty(raw_frame):
                continue
            filtered_frame = filter_fitness_reviews(raw_frame)
            raw_frames.append(filtered_frame)

            normalized_frame = self.normalizer.normalize_reviews(
                filtered_frame,
                source_name=self._source_name(source),
                source_type=source.source_type,
            )
            if self._is_empty(normalized_frame):
                continue
            normalized_frames.append(normalized_frame)

        merged_raw = self.merge(raw_frames)
        self.registry.save_dataframe("data/raw/merged_raw.parquet", merged_raw)
        if not normalized_frames:
            return self.normalizer.normalize_reviews([], source_name=self.name, source_type="collect")

        return self.merge(normalized_frames)

    def merge(self, frames: list[Any]) -> Any:
        """Merge frame-like objects and remove safe duplicates."""

        records: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()

        for frame in frames:
            for row in self._to_records(frame):
                fingerprint = json.dumps(row, sort_keys=True, default=str)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                records.append(row)

        return self._build_frame(records)

    def _collect_source(self, source: SourceCandidate) -> Any:
        """Collect one source using the appropriate local provider stub or loader."""

        if source.source_type == "hf_dataset":
            dataset = self.hf_loader.load(source.uri)
            return self.hf_loader.to_dataframe(dataset)

        if source.source_type == "scrape":
            html = self._load_html(source)
            if not html:
                return self._empty_frame()
            return self.scraper(html)

        if source.source_type == "github_repo":
            self.logger.warning("Skipping github_repo source during collect stage: %s", source.uri)
            return self._empty_frame()

        if source.source_type == "api":
            self.logger.warning("Skipping api source during collect stage: %s", source.uri)
            return self._empty_frame()

        self.logger.warning("Skipping unsupported source_type=%s", source.source_type)
        return self._empty_frame()

    def _load_html(self, source: SourceCandidate) -> str:
        """Load local HTML from metadata or from a filesystem path in the URI."""

        if isinstance(source.metadata, dict) and isinstance(source.metadata.get("html"), str):
            return source.metadata["html"]

        candidate_path = Path(source.uri)
        if candidate_path.exists():
            return candidate_path.read_text(encoding="utf-8")

        return ""

    def _source_name(self, source: SourceCandidate) -> str:
        """Resolve a stable, human-readable source name for provenance tracking."""

        if source.title:
            return source.title
        if source.source_id:
            return source.source_id
        return source.uri

    def _to_records(self, frame: Any) -> list[dict[str, Any]]:
        """Materialize a frame-like object into row dictionaries."""

        if hasattr(frame, "to_dict"):
            try:
                records = frame.to_dict(orient="records")
            except TypeError:
                records = frame.to_dict()
            if isinstance(records, list):
                return [dict(row) for row in records]
            if isinstance(records, dict):
                columns = list(records.keys())
                row_count = len(records[columns[0]]) if columns else 0
                return [{column: records[column][index] for column in columns} for index in range(row_count)]
            return [dict(row) for row in records]

        if isinstance(frame, list):
            return [dict(row) for row in frame]

        return []

    def _build_frame(self, records: list[dict[str, Any]]) -> Any:
        """Return a dataframe-like object using the local pandas-compatible fallback."""

        try:
            import pandas as pd  # type: ignore[import-not-found]

            return pd.DataFrame(records)
        except Exception:
            return _SimpleFrame(records)

    def _empty_frame(self) -> Any:
        """Return an empty frame-like object with no rows."""

        return self._build_frame([])

    def _is_empty(self, frame: Any) -> bool:
        """Check whether a frame-like object is empty."""

        if frame is None:
            return True
        return bool(getattr(frame, "empty", False))


class _SimpleFrame:
    """Fallback dataframe-like object used when pandas is unavailable."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records = [dict(row) for row in (records or [])]
        self._columns = list(self._records[0].keys()) if self._records else []

    @property
    def empty(self) -> bool:
        return not self._records

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        if orient != "records":
            raise ValueError("Only records orientation is supported")
        if not self._columns:
            return [dict(row) for row in self._records]
        return [{column: row.get(column) for column in self._columns} for row in self._records]
