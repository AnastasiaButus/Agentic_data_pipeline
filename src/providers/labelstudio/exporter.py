"""Label Studio export helpers for annotation tasks."""

from __future__ import annotations

from typing import Any


CANONICAL_EXPORT_COLUMNS = (
    "id",
    "source",
    "text",
    "label",
    "rating",
    "created_at",
    "split",
    "meta_json",
    "sentiment_label",
    "effect_label",
    "confidence",
)


def to_labelstudio_tasks(df: Any) -> list[dict[str, Any]]:
    """Convert labeled review rows into Label Studio-compatible task dictionaries."""

    rows = _to_records(df)
    tasks: list[dict[str, Any]] = []

    for row in rows:
        data_payload = {column: row.get(column) for column in CANONICAL_EXPORT_COLUMNS}
        tasks.append(
            {
                "data": data_payload,
                "predictions": [
                    {
                        "model_version": "annotation_agent",
                        "score": _coerce_confidence(row.get("confidence")),
                        "result": [
                            _build_choice_result("sentiment_label", "text", row.get("sentiment_label") or "other"),
                            _build_choice_result("effect_label", "text", row.get("effect_label") or "other"),
                        ],
                    }
                ],
            }
        )

    return tasks


def _build_choice_result(from_name: str, to_name: str, label: Any) -> dict[str, Any]:
    """Build one Label Studio choice prediction entry."""

    return {
        "from_name": from_name,
        "to_name": to_name,
        "type": "choices",
        "value": {"choices": [str(label)]},
    }


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


def _coerce_confidence(value: Any) -> float:
    """Clamp confidence into the closed interval [0, 1]."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0

    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric