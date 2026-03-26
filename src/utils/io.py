"""Small file IO helpers used by the infrastructure layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a target path and return the normalized path."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON content to disk with a stable UTF-8 encoding."""

    target_path = ensure_parent(path)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target_path


def read_json(path: str | Path) -> Any:
    """Read JSON content from disk."""

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str | Path, text: str) -> Path:
    """Write plain text content to disk."""

    target_path = ensure_parent(path)
    target_path.write_text(text, encoding="utf-8")
    return target_path


def read_text(path: str | Path) -> str:
    """Read plain text content from disk."""

    source_path = Path(path)
    return source_path.read_text(encoding="utf-8")
