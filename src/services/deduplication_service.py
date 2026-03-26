"""Deduplication helpers for the quality stage."""

from __future__ import annotations

from typing import Any

from src.utils.text_cleaning import normalize_whitespace


def drop_duplicates_by_text(df: Any) -> Any:
    """Drop duplicate rows using the canonical text field as the key."""

    records = _to_records(df)
    if not records:
        return _build_like(df, records)

    seen: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for row in records:
        key = normalize_whitespace(row.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)

    return _build_like(df, deduplicated)


def _to_records(df: Any) -> list[dict[str, Any]]:
    """Materialize a dataframe-like object into row dictionaries."""

    if hasattr(df, "to_dict"):
        try:
            records = df.to_dict(orient="records")
        except TypeError:
            records = df.to_dict()
        if isinstance(records, list):
            return [dict(row) for row in records]
        if isinstance(records, dict):
            columns = list(records.keys())
            row_count = len(records[columns[0]]) if columns else 0
            return [{column: records[column][index] for column in columns} for index in range(row_count)]
        return [dict(row) for row in records]

    if isinstance(df, list):
        return [dict(row) for row in df]

    return []


def _build_like(original: Any, rows: list[dict[str, Any]]) -> Any:
    """Return a dataframe-like object that matches the input style as closely as possible."""

    columns = list(getattr(original, "columns", []))
    if rows and not columns:
        columns = list(rows[0].keys())

    try:
        import pandas as pd  # type: ignore[import-not-found]

        return pd.DataFrame(rows, columns=columns or None)
    except Exception:
        return _SimpleFrame(rows, columns)


class _SimpleFrame:
    """Fallback dataframe-like object used when pandas is unavailable."""

    def __init__(self, records: list[dict[str, Any]] | None = None, columns: list[str] | None = None) -> None:
        self._records = [dict(row) for row in (records or [])]
        self._columns = list(columns or [])
        if not self._columns and self._records:
            self._columns = list(self._records[0].keys())

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