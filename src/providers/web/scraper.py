"""Local HTML review parsing helpers used by the collect stage."""

from __future__ import annotations

import re
from html import unescape
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    class _RequestsShim:
        """Fallback shim so tests can monkeypatch requests.get without the dependency."""

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("requests is required to fetch remote HTML pages")

    requests = _RequestsShim()

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None  # type: ignore[assignment]

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


def scrape_url(
    url: str,
    selector: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    html: str | None = None,
) -> Any:
    """Fetch a page and extract rows from the given CSS selector."""

    cleaned_selector = str(selector or "").strip()
    if not cleaned_selector:
        return pd.DataFrame([])

    page_html = html if html is not None else _fetch_html(url, headers=headers, timeout=timeout)
    return parse_selector_blocks(page_html, cleaned_selector)


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


def parse_selector_blocks(html: str, selector: str) -> Any:
    """Parse arbitrary selected HTML blocks into a dataframe-like object."""

    cleaned_selector = str(selector or "").strip()
    if not cleaned_selector:
        return pd.DataFrame([])

    if BeautifulSoup is not None:
        return _parse_selector_blocks_with_bs4(html, cleaned_selector)
    return _parse_selector_blocks_with_regex(html, cleaned_selector)


def _fetch_html(url: str, *, headers: dict[str, str] | None = None, timeout: float | None = None) -> str:
    """Fetch HTML for a remote page using requests."""

    cleaned_url = str(url or "").strip()
    if not cleaned_url:
        raise ValueError("url must not be empty")

    response = requests.get(
        cleaned_url,
        headers=dict(headers or {}),
        timeout=timeout if timeout is not None else 20,
    )
    response.raise_for_status()
    return getattr(response, "text", "")


def _parse_selector_blocks_with_bs4(html: str, selector: str) -> Any:
    """Use BeautifulSoup CSS selectors when the dependency is available."""

    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []

    for element in soup.select(selector):
        record = _record_from_element(element)
        if record is not None:
            records.append(record)

    return pd.DataFrame(records)


def _parse_selector_blocks_with_regex(html: str, selector: str) -> Any:
    """Fallback parser for simple selectors when BeautifulSoup is unavailable."""

    tag_name, class_name, element_id = _parse_simple_selector(selector)
    if tag_name is None and class_name is None and element_id is None:
        return pd.DataFrame([])

    tag_pattern = re.escape(tag_name) if tag_name else r"[a-zA-Z0-9]+"
    class_fragment = ""
    id_fragment = ""
    if class_name is not None:
        escaped_class = re.escape(class_name)
        class_fragment = rf'(?=[^>]*\bclass=["\'][^"\']*\b{escaped_class}\b[^"\']*["\'])'
    if element_id is not None:
        escaped_id = re.escape(element_id)
        id_fragment = rf'(?=[^>]*\bid=["\']{escaped_id}["\'])'

    pattern = re.compile(
        rf"<({tag_pattern})\b{class_fragment}{id_fragment}([^>]*)>(.*?)</\1>",
        re.IGNORECASE | re.DOTALL,
    )
    records: list[dict[str, Any]] = []

    for match in pattern.finditer(html):
        attributes = match.group(2)
        body = match.group(3)

        if class_name is not None:
            class_value = _extract_attribute(attributes, "class") or ""
            classes = {part.strip() for part in class_value.split() if part.strip()}
            if class_name not in classes:
                continue

        if element_id is not None:
            if (_extract_attribute(attributes, "id") or "") != element_id:
                continue

        record = _record_from_tag(attributes, body)
        if record is not None:
            records.append(record)

    return pd.DataFrame(records)


def _parse_simple_selector(selector: str) -> tuple[str | None, str | None, str | None]:
    """Support a safe subset of CSS selectors for the stdlib fallback path."""

    cleaned = str(selector or "").strip()
    if not cleaned or " " in cleaned or ">" in cleaned or "[" in cleaned or ":" in cleaned:
        return None, None, None
    original = cleaned

    tag_name: str | None = None
    class_name: str | None = None
    element_id: str | None = None

    if "#" in cleaned:
        left, right = cleaned.split("#", 1)
        if not right:
            return None, None, None
        element_id = right.strip() or None
        cleaned = left.strip()

    if "." in cleaned:
        left, right = cleaned.split(".", 1)
        if not right:
            return None, None, None
        tag_name = left.strip() or None
        class_name = right.strip() or None
        return tag_name, class_name, element_id

    if original.startswith("."):
        class_name = cleaned[1:].strip()
        return None, class_name or None, element_id

    if original.startswith("#"):
        return None, None, element_id or None

    return cleaned or None, None, element_id


def _record_from_element(element: Any) -> dict[str, Any] | None:
    """Build one raw review-like record from a BeautifulSoup element."""

    text_value = _normalize_html_text(element.get("data-text")) or _normalize_html_text(element.get_text(" ", strip=True))
    if not text_value:
        return None

    body_text = _normalize_html_text(element.get_text(" ", strip=True))
    record: dict[str, Any] = {"text": text_value}

    rating_value = element.get("data-rating") or element.get("data-score")
    if rating_value is not None:
        record["rating"] = _coerce_rating(str(rating_value))

    product_name = element.get("data-product") or element.get("data-product-name")
    category = element.get("data-category")
    title = element.get("data-title")

    if product_name is not None:
        record["product_name"] = _normalize_html_text(product_name)
    if category is not None:
        record["category"] = _normalize_html_text(category)
    if title is not None:
        record["title"] = _normalize_html_text(title)
    if body_text:
        record["content"] = body_text

    return record


def _record_from_tag(attributes: str, body: str) -> dict[str, Any] | None:
    """Build one raw review-like record from a regex-captured HTML fragment."""

    data_text = _normalize_html_text(_extract_attribute(attributes, "data-text"))
    body_text = _normalize_html_text(re.sub(r"<[^>]+>", " ", body))
    text_value = data_text or body_text
    if not text_value:
        return None

    record: dict[str, Any] = {"text": text_value}

    rating_value = _extract_attribute(attributes, "data-rating") or _extract_attribute(attributes, "data-score")
    if rating_value is not None:
        record["rating"] = _coerce_rating(rating_value)

    product_name = _extract_attribute(attributes, "data-product") or _extract_attribute(attributes, "data-product-name")
    category = _extract_attribute(attributes, "data-category")
    title = _extract_attribute(attributes, "data-title")

    if product_name is not None:
        record["product_name"] = _normalize_html_text(product_name)
    if category is not None:
        record["category"] = _normalize_html_text(category)
    if title is not None:
        record["title"] = _normalize_html_text(title)
    if body_text:
        record["content"] = body_text

    return record


def _extract_attribute(attributes: str, name: str) -> str | None:
    """Extract a single HTML attribute from a review block tag."""

    match = re.search(rf'{name}=["\']([^"\']+)["\']', attributes, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _normalize_html_text(value: Any) -> str:
    """Normalize arbitrary HTML-derived text into a compact string."""

    if value is None:
        return ""
    return " ".join(unescape(str(value)).split())


def _coerce_rating(value: str) -> int | float | str:
    """Convert rating strings into numeric values when possible."""

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
