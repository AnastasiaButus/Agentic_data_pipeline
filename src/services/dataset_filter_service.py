"""Lightweight dataset filtering helpers for the collect stage."""

from __future__ import annotations

from typing import Any


def filter_fitness_reviews(df: Any) -> Any:
    """Apply a soft fitness-related filter when descriptive columns are present."""

    rows = _to_records(df)
    if not rows:
        return _build_like(df, rows)

    candidate_rows = []
    for row in rows:
        text_blob = " ".join(
            str(row.get(field, "")) for field in ("product_name", "category", "title", "text", "content")
        ).lower()
        if not text_blob:
            candidate_rows.append(row)
            continue
        if any(keyword in text_blob for keyword in ("fitness", "supplement", "protein", "review")):
            candidate_rows.append(row)

    return _build_like(df, candidate_rows or rows)


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

    columns = list(rows[0].keys()) if rows else list(getattr(original, "columns", []))

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
