"""Local HTML review parsing helpers used by the collect stage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Any


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
            return [{column: row.get(column) for column in self._columns} for row in self._records]

    class _PandasShim:
        DataFrame = _SimpleDataFrame

    pd = _PandasShim()


def parse_review_blocks(html: str) -> Any:
    """Parse review blocks from local HTML into a dataframe-like object."""

    records: list[dict[str, Any]] = []
    pattern = re.compile(r'<div\s+class=["\']review["\']([^>]*)>(.*?)</div>', re.IGNORECASE | re.DOTALL)

    for match in pattern.finditer(html):
        attributes = match.group(1)
        body = match.group(2)
        data_text = _extract_attribute(attributes, "data-text")
        data_rating = _extract_attribute(attributes, "data-rating")
        if data_text is None or data_rating is None:
            continue

        record: dict[str, Any] = {
            "text": unescape(data_text),
            "rating": _coerce_rating(data_rating),
        }
        product_name = _extract_attribute(attributes, "data-product")
        category = _extract_attribute(attributes, "data-category")
        title = _extract_attribute(attributes, "data-title")
        if product_name is not None:
            record["product_name"] = unescape(product_name)
        if category is not None:
            record["category"] = unescape(category)
        if title is not None:
            record["title"] = unescape(title)

        body_text = re.sub(r"<[^>]+>", " ", body)
        body_text = " ".join(body_text.split())
        if body_text:
            record.setdefault("content", body_text)

        records.append(record)

    return pd.DataFrame(records)


def _extract_attribute(attributes: str, name: str) -> str | None:
    """Extract a single HTML attribute from a review block tag."""

    match = re.search(rf'{name}=["\']([^"\']+)["\']', attributes, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _coerce_rating(value: str) -> int | float | str:
    """Convert rating strings into numeric values when possible."""

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
