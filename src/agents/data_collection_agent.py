"""Collect raw data from configured sources and normalize it into the canonical schema."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.constants import STANDARD_COLUMNS
from src.domain import SourceCandidate
from src.providers.apis.json_api_client import JsonAPIClient
from src.providers.datasets.hf_loader import HFDatasetLoader
from src.providers.web.scraper import parse_review_blocks, scrape_url
from src.services.artifact_registry import ArtifactRegistry
from src.services.dataset_filter_service import filter_topic_rows
from src.services.schema_normalization_service import SchemaNormalizationService


class DataCollectionAgent(BaseAgent):
    """Collect and normalize source data without introducing a new architecture layer."""

    def __init__(
        self,
        ctx: Any,
        hf_loader: HFDatasetLoader | None = None,
        api_client: Any | None = None,
        web_scraper: Any | None = None,
        normalizer: SchemaNormalizationService | None = None,
        scraper: Any | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        """Bind the agent to the pipeline context and its helpers."""

        super().__init__(ctx, registry if registry is not None else ArtifactRegistry(ctx))
        self.hf_loader = hf_loader if hf_loader is not None else HFDatasetLoader()
        self.api_client = api_client if api_client is not None else JsonAPIClient()
        self.web_scraper = web_scraper if web_scraper is not None else scrape_url
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
            filtered_frame = filter_topic_rows(
                raw_frame,
                topic=self._request_topic(),
                domain=self._request_domain(),
            )
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

    def fetch_api(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        method: str = "GET",
        json_payload: Any | None = None,
        data_payload: Any | None = None,
        timeout: float | None = None,
        records_path: str = "",
        field_map: Mapping[str, Any] | None = None,
    ) -> Any:
        """Fetch tabular rows from a JSON API and return a dataframe-like object."""

        cleaned_endpoint = str(endpoint or "").strip()
        if not cleaned_endpoint:
            return self._empty_frame()

        try:
            cleaned_method = str(method or "GET").strip().upper() or "GET"
            if cleaned_method == "GET":
                payload = self.api_client.fetch_json(
                    cleaned_endpoint,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                response = self.api_client.request(
                    cleaned_method,
                    cleaned_endpoint,
                    params=params,
                    headers=headers,
                    json=json_payload,
                    data=data_payload,
                    timeout=timeout,
                )
                payload = response.json() if hasattr(response, "json") else response
        except Exception:
            self.logger.warning("Skipping api source during collect stage: %s", cleaned_endpoint)
            return self._empty_frame()

        records = self._extract_api_records(payload, records_path=records_path)
        mapped_records = self._apply_api_field_map(records, field_map=field_map)
        if not mapped_records:
            return self._empty_frame()
        return self._build_frame(mapped_records)

    def scrape(
        self,
        url: str,
        selector: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        html: str | None = None,
    ) -> Any:
        """Fetch or parse HTML rows from a page using a CSS selector."""

        cleaned_selector = str(selector or "").strip()
        if not cleaned_selector:
            return self._empty_frame()
        return self.web_scraper(
            str(url or "").strip(),
            cleaned_selector,
            headers=dict(headers or {}),
            timeout=timeout,
            html=html,
        )

    def _collect_source(self, source: SourceCandidate) -> Any:
        """Collect one source using the appropriate local provider stub or loader."""

        if source.source_type == "hf_dataset":
            try:
                dataset = self.hf_loader.load(source.uri)
                return self.hf_loader.to_dataframe(dataset)
            except Exception:
                self.logger.warning("Skipping hf_dataset source during collect stage: %s", source.uri)
                return self._empty_frame()

        if source.source_type == "scrape":
            html = self._load_html(source)
            metadata = source.metadata if isinstance(source.metadata, dict) else {}
            selector = str(metadata.get("selector") or "").strip()
            headers = self._metadata_string_mapping(metadata.get("headers"))
            timeout = self._metadata_timeout(metadata.get("timeout"))
            try:
                if selector:
                    return self.scrape(
                        str(metadata.get("url") or source.uri),
                        selector,
                        headers=headers,
                        timeout=timeout,
                        html=html or None,
                    )
                if html:
                    return self.scraper(html)
            except Exception:
                self.logger.warning("Skipping scrape source during collect stage: %s", source.uri)
                return self._empty_frame()
            self.logger.warning("Skipping scrape source without selector or local HTML: %s", source.uri)
            return self._empty_frame()

        if source.source_type == "github_repo":
            self.logger.warning("Skipping github_repo source during collect stage: %s", source.uri)
            return self._empty_frame()

        if source.source_type == "api":
            metadata = source.metadata if isinstance(source.metadata, dict) else {}
            return self.fetch_api(
                str(metadata.get("endpoint") or source.uri),
                params=self._metadata_mapping(metadata.get("params")),
                headers=self._metadata_string_mapping(metadata.get("headers")),
                method=str(metadata.get("method") or "GET"),
                json_payload=metadata.get("json"),
                data_payload=metadata.get("data"),
                timeout=self._metadata_timeout(metadata.get("timeout")),
                records_path=str(metadata.get("records_path") or ""),
                field_map=self._metadata_mapping(metadata.get("field_map")),
            )

        self.logger.warning("Skipping unsupported source_type=%s", source.source_type)
        return self._empty_frame()

    def _extract_api_records(self, payload: Any, *, records_path: str = "") -> list[dict[str, Any]]:
        """Extract a list of row dictionaries from a JSON payload."""

        target = self._resolve_json_path(payload, records_path) if records_path else payload
        if isinstance(target, list):
            return [dict(row) for row in target if isinstance(row, dict)]

        if isinstance(target, dict):
            for key in ("items", "results", "records", "reviews", "data"):
                nested = target.get(key)
                if isinstance(nested, list):
                    return [dict(row) for row in nested if isinstance(row, dict)]
            return [dict(target)]

        return []

    def _apply_api_field_map(
        self,
        records: list[dict[str, Any]],
        *,
        field_map: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Overlay canonical keys onto raw API rows using a simple source-field map."""

        mapping = dict(field_map or {})
        if not mapping:
            return [dict(row) for row in records]

        output_rows: list[dict[str, Any]] = []
        for row in records:
            mapped_row = dict(row)
            for target_field, source_field in mapping.items():
                resolved_value = self._resolve_json_path(row, str(source_field or ""))
                if resolved_value is not None:
                    mapped_row[str(target_field)] = resolved_value
            output_rows.append(mapped_row)
        return output_rows

    def _resolve_json_path(self, payload: Any, path: str) -> Any:
        """Resolve a dotted path inside nested dict/list API payloads."""

        cleaned_path = str(path or "").strip()
        if not cleaned_path:
            return payload

        current = payload
        for part in cleaned_path.split("."):
            key = part.strip()
            if not key:
                return None
            if isinstance(current, dict):
                current = current.get(key)
                continue
            if isinstance(current, list) and key.isdigit():
                index = int(key)
                if index < 0 or index >= len(current):
                    return None
                current = current[index]
                continue
            return None
        return current

    def _metadata_mapping(self, value: Any) -> Mapping[str, Any] | None:
        """Return dict-like metadata only when the value is mapping-shaped."""

        return value if isinstance(value, Mapping) else None

    def _metadata_string_mapping(self, value: Any) -> dict[str, str] | None:
        """Normalize header-like metadata into string keys and values."""

        if not isinstance(value, Mapping):
            return None
        return {str(key): str(item) for key, item in value.items()}

    def _metadata_timeout(self, value: Any) -> float | None:
        """Coerce timeout metadata into a numeric value when possible."""

        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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

    def _request_topic(self) -> str:
        """Return the active request topic for soft collection-time filtering."""

        request = getattr(getattr(self.ctx, "config", None), "request", None)
        return str(getattr(request, "topic", "") or "").strip()

    def _request_domain(self) -> str:
        """Return the active request domain for soft collection-time filtering."""

        request = getattr(getattr(self.ctx, "config", None), "request", None)
        return str(getattr(request, "domain", "") or "").strip()

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
