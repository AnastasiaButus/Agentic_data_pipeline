"""Quality stage agent for detecting, fixing, and comparing data issues."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.constants import STANDARD_COLUMNS
from src.domain import QualityReport
from src.services.artifact_registry import ArtifactRegistry
from src.services.deduplication_service import drop_duplicates_by_text
from src.utils.text_cleaning import normalize_whitespace, safe_word_count


class DataQualityAgent(BaseAgent):
    """Detect, repair, and compare canonical review data."""

    def __init__(self, ctx: Any, registry: ArtifactRegistry | None = None) -> None:
        """Bind the agent to the execution context and artifact registry."""

        super().__init__(ctx, registry if registry is not None else ArtifactRegistry(ctx))

    def run(self, df: Any, strategy: dict[str, Any] | None = None) -> Any:
        """Execute the quality stage and persist cleaned and comparison artifacts."""

        applied_strategy = strategy or {
            "drop_empty_text": True,
            "min_words": 1,
            "normalize_whitespace": True,
            "duplicates": "drop",
            "outliers": "remove_iqr",
        }
        cleaned = self.fix(df, applied_strategy)
        comparison = self.compare(df, cleaned)
        self.registry.save_dataframe("data/interim/cleaned_v1.parquet", cleaned)
        self.registry.save_dataframe("data/interim/quality_compare.csv", comparison)
        return cleaned

    def detect_issues(self, df: Any) -> QualityReport:
        """Detect missing data, duplicates, outliers, imbalance, and warnings."""

        rows = self._to_records(df)
        row_count = len(rows)
        missing = self._missing_percentages(rows)
        duplicates = self._duplicate_count(rows)
        outliers = self._text_length_outliers(rows)
        imbalance = self._label_imbalance(rows)

        warnings: list[str] = []
        if row_count == 0:
            warnings.append("empty dataset")
        if any(value > 0 for value in missing.values()):
            warnings.append("missing values detected")
        if duplicates:
            warnings.append(f"duplicate text detected: {duplicates}")
        if outliers.get("text", {}).get("count", 0):
            warnings.append("text length outliers detected")
        if imbalance:
            warnings.append("label imbalance detected")

        return QualityReport(
            missing=missing,
            duplicates=duplicates,
            outliers=outliers,
            imbalance=imbalance,
            warnings=warnings,
        )

    def fix(self, df: Any, strategy: dict[str, Any]) -> Any:
        """Apply deterministic quality repairs while preserving the canonical schema."""

        rows = self._to_records(df)
        if not rows:
            return self._build_frame([])

        working_rows = [self._canonicalize_row(row) for row in rows]

        if strategy.get("normalize_whitespace"):
            for row in working_rows:
                row["text"] = normalize_whitespace(row.get("text"))

        if strategy.get("drop_empty_text"):
            working_rows = [row for row in working_rows if normalize_whitespace(row.get("text"))]

        min_words = int(strategy.get("min_words", 0) or 0)
        if min_words > 0:
            working_rows = [row for row in working_rows if safe_word_count(row.get("text")) >= min_words]

        if strategy.get("duplicates") == "drop":
            working_rows = self._to_records(drop_duplicates_by_text(self._build_frame(working_rows)))

        outlier_strategy = strategy.get("outliers")
        if outlier_strategy in {"clip_iqr", "remove_iqr"}:
            working_rows = self._apply_outlier_strategy(working_rows, outlier_strategy)

        return self._build_frame(working_rows)

    def compare(self, before: Any, after: Any) -> Any:
        """Compare before and after datasets using a small metric table."""

        before_rows = self._to_records(before)
        after_rows = self._to_records(after)

        metrics = [
            ("n_rows", float(len(before_rows)), float(len(after_rows))),
            ("duplicates", float(self._duplicate_count(before_rows)), float(self._duplicate_count(after_rows))),
            ("empty_text", float(self._empty_text_count(before_rows)), float(self._empty_text_count(after_rows))),
            ("mean_words", float(self._mean_words(before_rows)), float(self._mean_words(after_rows))),
        ]

        rows = [{"metric": metric, "before": before_value, "after": after_value} for metric, before_value, after_value in metrics]
        return self._build_frame(rows, columns=["metric", "before", "after"])

    def _missing_percentages(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Compute missing percentages for the canonical schema columns."""

        if not rows:
            return {column: 0.0 for column in STANDARD_COLUMNS}

        row_count = len(rows)
        missing: dict[str, float] = {}
        for column in STANDARD_COLUMNS:
            if column not in rows[0] and all(column not in row for row in rows):
                missing[column] = 1.0
                continue
            count = sum(1 for row in rows if self._is_missing(row.get(column)))
            missing[column] = count / row_count
        return missing

    def _duplicate_count(self, rows: list[dict[str, Any]]) -> int:
        """Count duplicate text rows after normalizing whitespace."""

        counts = Counter(normalize_whitespace(row.get("text")) for row in rows)
        return sum(count - 1 for text, count in counts.items() if text and count > 1)

    def _empty_text_count(self, rows: list[dict[str, Any]]) -> int:
        """Count rows whose canonical text is empty or whitespace only."""

        return sum(1 for row in rows if not normalize_whitespace(row.get("text")))

    def _mean_words(self, rows: list[dict[str, Any]]) -> float:
        """Compute the average canonical text length in words."""

        if not rows:
            return 0.0
        lengths = [safe_word_count(row.get("text")) for row in rows]
        return sum(lengths) / len(lengths)

    def _text_length_outliers(self, rows: list[dict[str, Any]]) -> dict[str, object]:
        """Detect IQR outliers using the canonical text length."""

        lengths = [safe_word_count(row.get("text")) for row in rows if normalize_whitespace(row.get("text"))]
        if len(lengths) < 4:
            return {"text": {"count": 0, "lower_bound": None, "upper_bound": None}}

        lower_quartile = self._percentile(lengths, 0.25)
        upper_quartile = self._percentile(lengths, 0.75)
        iqr = upper_quartile - lower_quartile
        lower_bound = lower_quartile - 1.5 * iqr
        upper_bound = upper_quartile + 1.5 * iqr
        count = sum(1 for length in lengths if length < lower_bound or length > upper_bound)
        return {"text": {"count": count, "lower_bound": lower_bound, "upper_bound": upper_bound}}

    def _label_imbalance(self, rows: list[dict[str, Any]]) -> dict[str, object]:
        """Summarize label balance when non-empty labels are available."""

        labels = [normalize_whitespace(row.get("label")) for row in rows if normalize_whitespace(row.get("label"))]
        if not labels:
            return {}

        counts = Counter(labels)
        total = sum(counts.values())
        distribution = {label: count / total for label, count in counts.items()}
        majority_ratio = max(distribution.values())
        minority_ratio = min(distribution.values())

        if majority_ratio < 0.75:
            return {}

        return {
            "label": {
                "distribution": distribution,
                "majority_ratio": majority_ratio,
                "minority_ratio": minority_ratio,
            }
        }

    def _apply_outlier_strategy(self, rows: list[dict[str, Any]], outlier_strategy: str) -> list[dict[str, Any]]:
        """Apply the requested IQR strategy to the text column."""

        lengths = [safe_word_count(row.get("text")) for row in rows if normalize_whitespace(row.get("text"))]
        if len(lengths) < 4:
            return rows

        lower_quartile = self._percentile(lengths, 0.25)
        upper_quartile = self._percentile(lengths, 0.75)
        iqr = upper_quartile - lower_quartile
        lower_bound = lower_quartile - 1.5 * iqr
        upper_bound = upper_quartile + 1.5 * iqr

        if outlier_strategy == "remove_iqr":
            return [row for row in rows if lower_bound <= safe_word_count(row.get("text")) <= upper_bound]

        clipped_rows: list[dict[str, Any]] = []
        clip_limit = max(1, int(math.floor(upper_bound)))
        for row in rows:
            text = normalize_whitespace(row.get("text"))
            if safe_word_count(text) > clip_limit:
                row = dict(row)
                row["text"] = " ".join(text.split()[:clip_limit])
            clipped_rows.append(row)
        return clipped_rows

    def _canonicalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Project a row onto the canonical schema."""

        return {column: row.get(column) for column in STANDARD_COLUMNS}

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

    def _build_frame(self, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Any:
        """Return a dataframe-like object for the current environment."""

        selected_columns = columns or list(STANDARD_COLUMNS)
        try:
            import pandas as pd  # type: ignore[import-not-found]

            return pd.DataFrame(rows, columns=selected_columns or None)
        except Exception:
            return _SimpleFrame(rows, selected_columns)

    def _is_missing(self, value: Any) -> bool:
        """Check whether a value should count as missing."""

        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        try:
            return bool(value != value)
        except Exception:
            return False

    def _percentile(self, values: list[int], percentile: float) -> float:
        """Compute a simple percentile without depending on numpy."""

        ordered = sorted(values)
        if not ordered:
            return 0.0
        if len(ordered) == 1:
            return float(ordered[0])

        position = (len(ordered) - 1) * percentile
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        if lower_index == upper_index:
            return float(ordered[lower_index])

        lower_value = ordered[lower_index]
        upper_value = ordered[upper_index]
        fraction = position - lower_index
        return float(lower_value + (upper_value - lower_value) * fraction)


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