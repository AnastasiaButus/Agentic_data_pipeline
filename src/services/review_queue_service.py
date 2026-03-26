"""Human review queue helpers for confidence-driven effect-label correction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.constants import STANDARD_COLUMNS
from src.core.context import PipelineContext
from src.core.exceptions import ValidationError
from src.services.artifact_registry import ArtifactRegistry


ANNOTATION_COLUMNS = ("sentiment_label", "effect_label", "confidence")
REVIEW_COLUMNS = ("reviewed_effect_label", "review_comment", "human_verified")
REVIEW_QUEUE_PATH = "data/interim/review_queue.csv"
CORRECTED_QUEUE_PATH = "data/interim/review_queue_corrected.csv"


class ReviewQueueService:
    """Build, load, and merge confidence-based human review queues."""

    def __init__(self, ctx: PipelineContext, registry: ArtifactRegistry | None = None) -> None:
        """Bind the service to the active context and artifact registry."""

        self.ctx = ctx
        self.registry = registry if registry is not None else ArtifactRegistry(ctx)

    def export_low_confidence_queue(self, df: Any, threshold: float = 0.7) -> Any:
        """Export rows whose confidence is below the threshold or missing."""

        rows = self._to_records(df)
        queue_rows: list[dict[str, Any]] = []

        for row in rows:
            confidence = self._coerce_confidence(row.get("confidence"))
            if confidence is None or confidence < threshold:
                queue_rows.append(self._build_queue_row(row))

        queue_frame = self._build_frame(queue_rows, self._queue_columns())
        self.registry.save_dataframe(REVIEW_QUEUE_PATH, queue_frame)
        return queue_frame

    def load_corrected_queue(self, path: str | Path | None = None) -> Any:
        """Load the manually corrected review queue from the default or provided path."""

        target_path = path or CORRECTED_QUEUE_PATH
        if not self.registry.exists(target_path):
            raise FileNotFoundError(f"Corrected review queue not found: {target_path}")
        return self.registry.load_dataframe(target_path)

    def merge_reviewed_labels(self, original_df: Any, corrected_df: Any) -> Any:
        """Merge reviewed effect labels back into the original annotated dataset by canonical id."""

        original_rows = self._to_records(original_df)
        corrected_rows = self._to_records(corrected_df)

        if not original_rows:
            if not corrected_rows:
                return self._build_frame([], self._merged_columns())

            # Even with no original data, corrected rows must still be validated.
            self._validate_corrected_rows(corrected_rows, {})
            return self._build_frame([], self._merged_columns())

        original_by_id = {self._require_id(row, "original_df"): dict(row) for row in original_rows}
        self._validate_corrected_rows(corrected_rows, original_by_id)

        merged_rows = [dict(row) for row in original_rows]
        index_by_id = {row["id"]: index for index, row in enumerate(merged_rows)}

        for corrected_row in corrected_rows:
            row_id = self._require_id(corrected_row, "corrected_df")
            merged_row = dict(merged_rows[index_by_id[row_id]])
            reviewed_effect_label = self._normalize_text(corrected_row.get("reviewed_effect_label"))
            review_comment = self._normalize_text(corrected_row.get("review_comment"))
            human_verified = self._coerce_bool(corrected_row.get("human_verified"))

            merged_row["reviewed_effect_label"] = reviewed_effect_label
            merged_row["review_comment"] = review_comment

            if reviewed_effect_label:
                merged_row["effect_label"] = reviewed_effect_label
                merged_row["confidence"] = 1.0
                merged_row["human_verified"] = True
            else:
                merged_row["human_verified"] = self._coerce_bool(merged_row.get("human_verified"))

            merged_rows[index_by_id[row_id]] = merged_row

        return self._build_frame(merged_rows, self._merged_columns())

    def _build_queue_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Add human review columns on top of the canonical and annotation schema."""

        queue_row = {column: row.get(column) for column in self._merged_columns() if column not in REVIEW_COLUMNS}
        queue_row["reviewed_effect_label"] = ""
        queue_row["review_comment"] = ""
        queue_row["human_verified"] = False
        return queue_row

    def _queue_columns(self) -> list[str]:
        """Return the column order for review queue exports."""

        return list(STANDARD_COLUMNS) + list(ANNOTATION_COLUMNS) + list(REVIEW_COLUMNS)

    def _merged_columns(self) -> list[str]:
        """Return the column order used after merging reviewed labels back in."""

        return list(STANDARD_COLUMNS) + list(ANNOTATION_COLUMNS) + list(REVIEW_COLUMNS)

    def _validate_corrected_rows(self, corrected_rows: list[dict[str, Any]], original_by_id: dict[str, dict[str, Any]]) -> None:
        """Validate duplicate ids and unknown ids before applying corrections."""

        seen_ids: set[str] = set()
        for row in corrected_rows:
            row_id = self._require_id(row, "corrected_df")
            if row_id in seen_ids:
                raise ValidationError(f"Duplicate id in corrected_df: {row_id}")
            seen_ids.add(row_id)
            if row_id not in original_by_id:
                raise ValidationError(f"Corrected id not found in original_df: {row_id}")

    def _to_records(self, df: Any) -> list[dict[str, Any]]:
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

    def _build_frame(self, rows: list[dict[str, Any]], columns: list[str]) -> Any:
        """Return a dataframe-like object while preserving deterministic column order."""

        try:
            import pandas as pd  # type: ignore[import-not-found]

            return pd.DataFrame(rows, columns=columns or None)
        except Exception:
            return _SimpleFrame(rows, columns)

    def _coerce_confidence(self, value: Any) -> float | None:
        """Convert confidence to float and preserve missing or NaN values as queue candidates."""

        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric:
            return None
        return numeric

    def _coerce_bool(self, value: Any) -> bool:
        """Normalize review flags without surprising truthy string behavior."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    def _normalize_text(self, value: Any) -> str:
        """Normalize review text values and treat blank strings as empty corrections."""

        if value is None:
            return ""
        return str(value).strip()

    def _require_id(self, row: dict[str, Any], source_name: str) -> str:
        """Read the canonical id field and fail clearly when it is missing."""

        row_id = self._normalize_text(row.get("id"))
        if not row_id:
            raise ValidationError(f"Missing id in {source_name}")
        return row_id


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