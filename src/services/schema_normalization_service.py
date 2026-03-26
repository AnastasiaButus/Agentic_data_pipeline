"""Normalize raw review rows into the shared pipeline schema."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, Iterable

from src.core.constants import STANDARD_COLUMNS
from src.core.exceptions import ValidationError


try:
    import pandas as pd
except ModuleNotFoundError:
    class _SimpleDataFrame:
        """Minimal dataframe-like object used when pandas is unavailable."""

        def __init__(self, records: list[dict[str, Any]] | None = None, columns: list[str] | None = None) -> None:
            self._records = [dict(row) for row in (records or [])]
            self._columns = list(columns or [])
            if not self._columns and self._records:
                self._columns = list(self._records[0].keys())

        @property
        def empty(self) -> bool:
            """Return whether the frame contains any rows."""

            return not self._records

        @property
        def columns(self) -> list[str]:
            """Return the frame columns in order."""

            return list(self._columns)

        def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
            """Return a records-oriented representation."""

            if orient != "records":
                raise ValueError("Only records orientation is supported")
            if not self._columns:
                return [dict(row) for row in self._records]

            filtered_rows: list[dict[str, Any]] = []
            for row in self._records:
                filtered_rows.append({column: row.get(column) for column in self._columns})
            return filtered_rows

    class _PandasShim:
        """Fallback namespace exposing the DataFrame constructor used by this service."""

        DataFrame = _SimpleDataFrame

    pd = _PandasShim()


class SchemaNormalizationService:
    """Normalize raw review data into the canonical review schema."""

    def normalize_reviews(self, df: Any, source_name: str, source_type: str) -> Any:
        """Normalize reviews and preserve useful raw fields in `meta_json`."""

        rows = self._rows_from_input(df)
        if not rows:
            return pd.DataFrame([], columns=list(STANDARD_COLUMNS))

        if not any(self._has_content_column(row) for row in rows):
            raise ValidationError("normalize_reviews requires text, content, or review_text")

        normalized_rows = [self._normalize_row(row, source_name, source_type) for row in rows]
        return pd.DataFrame(normalized_rows, columns=list(STANDARD_COLUMNS))

    def _rows_from_input(self, df: Any) -> list[dict[str, Any]]:
        """Materialize rows from list-like or dataframe-like inputs."""

        if hasattr(df, "to_dict"):
            try:
                records = df.to_dict(orient="records")
            except TypeError:
                records = df.to_dict()

            if isinstance(records, list):
                return [dict(row) for row in records]
            if isinstance(records, Mapping):
                columns = list(records.keys())
                first_column = records[columns[0]] if columns else []
                row_count = len(first_column) if hasattr(first_column, "__len__") else 0
                rows: list[dict[str, Any]] = []
                for index in range(row_count):
                    rows.append({column: records[column][index] for column in columns})
                return rows
            return [dict(row) for row in records]

        if isinstance(df, dict):
            return [dict(df)]

        if isinstance(df, (str, bytes)):
            return []

        if isinstance(df, Iterable):
            return [dict(row) for row in df]

        return []

    def _has_content_column(self, row: dict[str, Any]) -> bool:
        """Check whether a row exposes one of the supported content columns."""

        return any(key in row for key in ("text", "content", "review_text"))

    def _normalize_row(self, row: dict[str, Any], source_name: str, source_type: str) -> dict[str, Any]:
        """Normalize a single review row while preserving unused fields in JSON form."""

        content = self._first_present(row, ("text", "content", "review_text"))
        if content is None:
            raise ValidationError("normalize_reviews requires text, content, or review_text")

        raw_label = row.get("raw_label")
        if raw_label is None and "label" in row:
            raw_label = row.get("label")

        collected_at = datetime.now(timezone.utc).isoformat()
        record_id = self._stable_record_id(source_name, content)
        rating = row.get("rating")
        split = row.get("split")
        language = row.get("language") or "unknown"
        product_name = row.get("product_name")
        confidence = None

        meta_payload = {
            key: value
            for key, value in row.items()
            if key not in {"text", "content", "review_text", "label", "raw_label", "rating", "split"}
        }
        meta_payload.update(
            {
                "record_id": record_id,
                "source_name": source_name,
                "source_type": source_type,
                "content": content,
                "raw_label": raw_label,
                "language": language,
                "product_name": product_name,
                "confidence": confidence,
            }
        )

        return {
            "id": record_id,
            "source": source_name,
            "text": content,
            "label": raw_label,
            "rating": rating,
            "created_at": collected_at,
            "split": split,
            "meta_json": json.dumps(meta_payload, ensure_ascii=False, sort_keys=True, default=str),
        }

    def _first_present(self, row: dict[str, Any], keys: tuple[str, ...]) -> Any:
        """Return the first present value from a key priority list."""

        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return None

    def _stable_record_id(self, source_name: str, content: Any) -> str:
        """Derive a stable identifier from content and source name."""

        digest_input = f"{source_name}\u0000{content}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()
