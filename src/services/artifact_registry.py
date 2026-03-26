"""Filesystem-backed artifact registry for pipeline outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.core.context import PipelineContext
from src.utils.io import ensure_parent, read_json, read_text, write_json, write_text


class ArtifactRegistry:
    """Resolve artifact paths relative to the project root and persist them."""

    def __init__(self, ctx: PipelineContext) -> None:
        """Bind the registry to the active pipeline context."""

        self.ctx = ctx
        self.root_dir = Path(ctx.paths.root_dir)

    def save_dataframe(self, path: str | Path, df: Any) -> Path:
        """Persist tabular data, preferring parquet and falling back to CSV."""

        target_path = self._resolve(path)
        records, columns = self._extract_records(df)

        if target_path.suffix.lower() in {".parquet", ".pq"}:
            try:
                self._write_parquet(target_path, df)
                return target_path
            except Exception:
                # CSV fallback keeps the registry usable when parquet support is missing.
                pass

        self._write_csv(target_path, records, columns)
        return target_path

    def load_dataframe(self, path: str | Path) -> Any:
        """Load tabular data from parquet if available, otherwise from CSV."""

        target_path = self._resolve(path)
        self._require_exists(target_path)

        if target_path.suffix.lower() in {".parquet", ".pq"}:
            try:
                return self._read_parquet(target_path)
            except Exception:
                # Fall back to the CSV payload written by save_dataframe.
                pass

        return self._read_csv(target_path)

    def save_json(self, path: str | Path, payload: Any) -> Path:
        """Persist JSON payloads under the project root."""

        return write_json(self._resolve(path), payload)

    def load_json(self, path: str | Path) -> Any:
        """Load a JSON payload and raise a clear error when it is missing."""

        target_path = self._resolve(path)
        self._require_exists(target_path)
        return read_json(target_path)

    def save_markdown(self, path: str | Path, text: str) -> Path:
        """Persist markdown content as plain UTF-8 text."""

        return write_text(self._resolve(path), text)

    def save_text(self, path: str | Path, text: str) -> Path:
        """Persist plain text content under the project root."""

        return write_text(self._resolve(path), text)

    def load_text(self, path: str | Path) -> str:
        """Load plain text content and raise a clear error when missing."""

        target_path = self._resolve(path)
        self._require_exists(target_path)
        return read_text(target_path)

    def exists(self, path: str | Path) -> bool:
        """Check whether a resolved artifact exists on disk."""

        return self._resolve(path).exists()

    def _resolve(self, path: str | Path) -> Path:
        """Resolve a user-supplied path relative to the project root."""

        candidate_path = Path(path)
        if candidate_path.is_absolute():
            candidate_path = Path(*candidate_path.parts[1:])
        return self.root_dir / candidate_path

    def _require_exists(self, path: Path) -> None:
        """Raise a clear, testable error for missing artifacts."""

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

    def _extract_records(self, df: Any) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert supported table-like inputs into row dictionaries."""

        if hasattr(df, "to_dict"):
            records = list(df.to_dict(orient="records"))
            columns = list(getattr(df, "columns", records[0].keys() if records else []))
            return records, columns

        records = [dict(row) for row in df]
        columns = list(records[0].keys()) if records else []
        return records, columns

    def _write_parquet(self, path: Path, df: Any) -> None:
        """Write parquet data when pandas and a parquet engine are available."""

        import pandas as pd  # type: ignore[import-not-found]

        if isinstance(df, pd.DataFrame):
            ensure_parent(path)
            df.to_parquet(path, index=False)
            return

        frame = pd.DataFrame(self._extract_records(df)[0])
        ensure_parent(path)
        frame.to_parquet(path, index=False)

    def _read_parquet(self, path: Path) -> Any:
        """Read parquet data when pandas support is available."""

        import pandas as pd  # type: ignore[import-not-found]

        return pd.read_parquet(path)

    def _write_csv(self, path: Path, records: list[dict[str, Any]], columns: list[str]) -> None:
        """Write a JSON-encoded CSV fallback that preserves basic Python types."""

        ensure_parent(path)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
            for record in records:
                writer.writerow({column: json.dumps(record.get(column)) for column in columns})

    def _read_csv(self, path: Path) -> Any:
        """Read the CSV fallback and reconstruct the original row dictionaries."""

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = [
                {column: json.loads(value) for column, value in row.items()}
                for row in reader
            ]

        try:
            import pandas as pd  # type: ignore[import-not-found]

            return pd.DataFrame(rows, columns=columns or None)
        except Exception:
            return rows
