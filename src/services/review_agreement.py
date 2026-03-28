"""Helpers for honest auto-vs-human agreement reporting on the reviewed subset."""

from __future__ import annotations

import math
from typing import Any

try:  # pragma: no cover - exercised when sklearn is installed in the runtime.
    from sklearn.metrics import cohen_kappa_score
except ImportError:  # pragma: no cover - fallback keeps local/offline tests runnable.
    cohen_kappa_score = None  # type: ignore[assignment]


def build_review_agreement_summary(annotated: Any, corrected_queue: Any) -> dict[str, Any]:
    """Compare auto effect labels against human-reviewed labels on the reviewed subset."""

    annotated_rows = _to_records(annotated)
    corrected_rows = _to_records(corrected_queue) if corrected_queue is not None else []
    annotated_by_id = {
        row_id: dict(row)
        for row in annotated_rows
        if (row_id := _normalize_id(row.get("id"))) is not None
    }

    auto_labels: list[str] = []
    human_labels: list[str] = []
    disagreement_examples: list[dict[str, Any]] = []
    n_valid_corrected_rows = 0
    n_reviewed_rows = 0
    n_missing_annotated_rows = 0
    n_missing_auto_labels = 0

    for row in corrected_rows:
        row_id = _normalize_id(row.get("id"))
        if row_id is None:
            continue

        n_valid_corrected_rows += 1
        human_label = _normalize_label(row.get("reviewed_effect_label"))
        if not human_label:
            continue

        n_reviewed_rows += 1
        source_row = annotated_by_id.get(row_id)
        if source_row is None:
            n_missing_annotated_rows += 1
            continue

        auto_label = _normalize_label(row.get("effect_label")) or _normalize_label(source_row.get("effect_label"))
        if not auto_label:
            n_missing_auto_labels += 1
            continue

        auto_labels.append(auto_label)
        human_labels.append(human_label)

        if auto_label != human_label and len(disagreement_examples) < 5:
            disagreement_examples.append(
                {
                    "id": row_id,
                    "source": _normalize_text(row.get("source")) or _normalize_text(source_row.get("source")),
                    "auto_effect_label": auto_label,
                    "reviewed_effect_label": human_label,
                    "confidence": _coerce_float(row.get("confidence"), source_row.get("confidence")),
                    "text_preview": _truncate_text(
                        _normalize_text(row.get("text")) or _normalize_text(source_row.get("text")),
                        limit=140,
                    ),
                }
            )

    compared_rows = len(human_labels)
    matched_rows = sum(1 for auto_label, human_label in zip(auto_labels, human_labels) if auto_label == human_label)
    disagreement_rows = compared_rows - matched_rows
    agreement = (matched_rows / compared_rows) if compared_rows else None
    kappa, kappa_status = _compute_kappa(auto_labels, human_labels)

    notes = [
        "This metric measures auto-vs-human agreement on the reviewed subset, not inter-reviewer agreement between two humans.",
    ]
    if corrected_queue is None:
        notes.append("Corrected queue is missing, so no reviewed subset is available for agreement analysis yet.")
    elif n_reviewed_rows == 0:
        notes.append("Corrected queue exists, but no reviewed_effect_label values were provided.")
    if n_missing_annotated_rows:
        notes.append("Some corrected ids were not found in the annotated dataset and were skipped.")
    if n_missing_auto_labels:
        notes.append("Some reviewed rows had no comparable auto effect_label and were skipped.")

    return {
        "comparison_scope": "auto_vs_human_reviewed_subset",
        "corrected_queue_found": corrected_queue is not None,
        "n_corrected_rows": n_valid_corrected_rows,
        "n_reviewed_rows": n_reviewed_rows,
        "compared_rows": compared_rows,
        "matched_rows": matched_rows,
        "disagreement_rows": disagreement_rows,
        "agreement": agreement,
        "kappa": kappa,
        "kappa_status": kappa_status,
        "n_missing_annotated_rows": n_missing_annotated_rows,
        "n_missing_auto_labels": n_missing_auto_labels,
        "auto_label_distribution": _count_labels(auto_labels),
        "human_label_distribution": _count_labels(human_labels),
        "disagreement_examples": disagreement_examples,
        "notes": notes,
    }


def _compute_kappa(auto_labels: list[str], human_labels: list[str]) -> tuple[float | None, str]:
    """Compute Cohen's kappa when the reviewed subset supports it."""

    if not human_labels:
        return None, "not_available_no_compared_rows"
    if len(human_labels) < 2:
        return None, "not_available_single_compared_row"
    if len(set(auto_labels) | set(human_labels)) < 2:
        return None, "not_available_single_label"
    if cohen_kappa_score is None:
        return None, "not_available_missing_dependency"

    try:
        value = float(cohen_kappa_score(human_labels, auto_labels))
    except Exception:
        return None, "not_available_error"
    if math.isnan(value):
        return None, "not_available_nan"
    return value, "computed"


def _count_labels(labels: list[str]) -> dict[str, int]:
    """Build a compact label distribution for agreement reporting."""

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def _to_records(df: Any) -> list[dict[str, Any]]:
    """Materialize dataframe-like inputs into row dictionaries."""

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


def _normalize_id(value: Any) -> str | None:
    """Normalize ids and treat blank values as missing."""

    normalized = _normalize_text(value)
    return normalized if normalized else None


def _normalize_label(value: Any) -> str:
    """Normalize effect-label values for stable comparison."""

    normalized = _normalize_text(value)
    if not normalized:
        return ""
    return normalized.lower().replace(" ", "_").replace("-", "_")


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary values into stable strings."""

    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(value: Any, *, limit: int) -> str:
    """Trim long previews so the report stays readable."""

    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _coerce_float(primary: Any, fallback: Any) -> float | None:
    """Read a confidence-like value from the corrected row or the original annotation row."""

    for value in (primary, fallback):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        return numeric
    return None
