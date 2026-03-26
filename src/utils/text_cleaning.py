"""Text cleaning helpers used by the quality stage."""

from __future__ import annotations


def normalize_whitespace(text: object | None) -> str:
    """Collapse repeated whitespace and return a safe string value."""

    if text is None:
        return ""
    return " ".join(str(text).split())


def safe_word_count(text: object | None) -> int:
    """Count words safely for None, blank strings, and normal text."""

    normalized = normalize_whitespace(text)
    if not normalized:
        return 0
    return len(normalized.split(" "))