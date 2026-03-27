"""Lightweight dataset filtering helpers for the collect stage."""

from __future__ import annotations

from typing import Any


def filter_topic_rows(
    df: Any,
    *,
    topic: str = "",
    domain: str = "",
    extra_keywords: list[str] | None = None,
) -> Any:
    """Apply a soft topic-aware filter when descriptive columns are present.

    The filter is intentionally conservative: if no rows match the inferred topic keywords,
    the original rows are returned unchanged so discovery/collection does not silently
    discard a usable dataset.
    """

    rows = _to_records(df)
    if not rows:
        return _build_like(df, rows)

    keywords = _build_topic_keywords(topic=topic, domain=domain, extra_keywords=extra_keywords)
    if not keywords:
        return _build_like(df, rows)

    candidate_rows = []
    for row in rows:
        text_blob = " ".join(
            str(row.get(field, "")) for field in ("product_name", "category", "title", "text", "content")
        ).lower()
        if not text_blob:
            candidate_rows.append(row)
            continue
        if any(keyword in text_blob for keyword in keywords):
            candidate_rows.append(row)

    return _build_like(df, candidate_rows or rows)


def filter_fitness_reviews(df: Any) -> Any:
    """Backward-compatible alias for the former fitness-specific helper."""

    return filter_topic_rows(df, topic="fitness supplements", domain="supplements")


def _build_topic_keywords(
    *,
    topic: str = "",
    domain: str = "",
    extra_keywords: list[str] | None = None,
) -> list[str]:
    """Extract stable, low-risk keywords from topic metadata for soft filtering."""

    raw_parts = [topic, domain, *(extra_keywords or [])]
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "review",
        "reviews",
        "dataset",
        "datasets",
        "classification",
        "instructions",
        "instruction",
        "text",
        "texts",
        "data",
    }

    keywords: list[str] = []
    for part in raw_parts:
        normalized = str(part or "").strip().lower().replace("-", " ").replace("_", " ")
        if not normalized:
            continue
        for token in normalized.split():
            cleaned = "".join(char for char in token if char.isalnum())
            if len(cleaned) < 4 or cleaned in stopwords:
                continue
            if cleaned not in keywords:
                keywords.append(cleaned)
            if cleaned.endswith("s") and len(cleaned) > 4:
                singular = cleaned[:-1]
                if singular not in stopwords and singular not in keywords:
                    keywords.append(singular)

    return keywords


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
