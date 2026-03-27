"""Annotation stage agent for deterministic spec generation, auto-labeling, and export prep."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.constants import STANDARD_COLUMNS
from src.domain import AnnotationSpec
from src.providers.labelstudio.exporter import to_labelstudio_tasks
from src.providers.labelstudio.validators import validate_labelstudio_tasks
from src.providers.llm.mock_llm import MockLLM
from src.services.artifact_registry import ArtifactRegistry


ANNOTATION_COLUMNS = ("sentiment_label", "effect_label", "confidence")
EFFECT_LABELS = ["energy", "side_effects", "other"]


class AnnotationAgent(BaseAgent):
    """Generate annotation specs, attach deterministic labels, and prepare review exports."""

    def __init__(
        self,
        ctx: Any,
        llm_client: Any | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        """Bind the agent to the runtime context and an optional mockable LLM client."""

        super().__init__(ctx, registry if registry is not None else ArtifactRegistry(ctx))
        self.llm_client = llm_client

    def map_rating_to_sentiment(self, rating: Any) -> str | None:
        """Map the canonical rating field to a deterministic sentiment label."""

        if rating is None:
            return None
        try:
            if rating != rating:  # NaN-safe check without introducing a hard dependency.
                return None
        except Exception:
            return None

        try:
            numeric_rating = float(rating)
        except (TypeError, ValueError):
            return None

        if numeric_rating <= 2:
            return "negative"
        if numeric_rating == 3:
            return "neutral"
        if numeric_rating >= 4:
            return "positive"
        return None

    def generate_spec(self, df: Any, task: str) -> str:
        """Render a deterministic markdown annotation spec for human review and future export."""

        effect_labels = self._effect_label_vocabulary()

        spec = AnnotationSpec(
            name=f"{self._slugify(task)}_annotation_spec",
            description=f"Annotation guide for {task} using the canonical text/id/label contract.",
            text_field="text",
            label_field="label",
            id_field="id",
            labels=["negative", "neutral", "positive", *effect_labels],
            instructions=[
                "Use the canonical text field for review text.",
                "Keep id stable and never invent new record identifiers.",
                "Annotate sentiment from rating when available, and effect from the review text.",
            ],
            output_format="markdown",
            metadata={"task": task, "field_contract": list(STANDARD_COLUMNS)},
        )
        _ = df  # The spec is deterministic for this step; examples are intentionally hand-crafted.
        return self._render_spec(spec)

    def auto_label(self, df: Any) -> Any:
        """Attach sentiment and effect labels while preserving the canonical review schema."""

        rows = self._to_records(df)
        output_rows: list[dict[str, Any]] = []
        effect_labels = self._effect_label_vocabulary()

        for row in rows:
            canonical_row = self._canonicalize_row(row)
            sentiment_label = self.map_rating_to_sentiment(canonical_row.get("rating"))
            effect_label, confidence = self._predict_effect(canonical_row.get("text"), effect_labels)
            canonical_row.update(
                {
                    "sentiment_label": sentiment_label,
                    "effect_label": effect_label,
                    "confidence": confidence,
                }
            )
            output_rows.append(canonical_row)

        return self._build_frame(output_rows)

    def check_quality(self, df_labeled: Any) -> dict[str, Any]:
        """Summarize the labeled batch for future HITL routing and confidence filtering."""

        rows = self._to_records(df_labeled)
        if not rows:
            return {"label_dist": {}, "confidence_mean": 0.0, "n_low_confidence": 0, "n_rows": 0}

        label_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        threshold = self._confidence_threshold()
        n_low_confidence = 0

        for row in rows:
            label = self._normalize_label(row.get("sentiment_label") or row.get("label") or "other")
            label_counts[label] = label_counts.get(label, 0) + 1

            confidence = self._coerce_confidence(row.get("confidence"))
            confidence_values.append(confidence)
            if confidence < threshold:
                n_low_confidence += 1

        total = sum(label_counts.values())
        label_dist = {label: count / total for label, count in label_counts.items()}
        confidence_mean = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        return {
            "label_dist": label_dist,
            "confidence_mean": confidence_mean,
            "n_low_confidence": n_low_confidence,
            "n_rows": len(rows),
            "confidence_threshold": threshold,
        }

    def export_to_labelstudio(self, df_labeled: Any) -> list[dict[str, Any]]:
        """Convert labeled rows into Label Studio-compatible task dictionaries."""

        tasks = to_labelstudio_tasks(df_labeled)
        validate_labelstudio_tasks(tasks)
        return tasks

    def _predict_effect(self, text: Any, effect_labels: list[str] | None = None) -> tuple[str, float]:
        """Predict an effect label using the optional LLM client or a deterministic fallback."""

        vocabulary = effect_labels if effect_labels is not None else self._effect_label_vocabulary()
        return self._predict_effect_with_vocab(text, vocabulary)

    def _predict_effect_with_vocab(self, text: Any, effect_labels: list[str]) -> tuple[str, float]:
        """Predict an effect label using the optional LLM client or a deterministic fallback."""

        if self.llm_client is not None and hasattr(self.llm_client, "classify_effect"):
            try:
                result = self.llm_client.classify_effect(str(text or ""), labels=effect_labels)
                label = self._normalize_label(getattr(result, "label", "other"))
                if label not in effect_labels:
                    label = self._fallback_effect_label(effect_labels)
                confidence = self._coerce_confidence(getattr(result, "confidence", 0.5))
                return label, confidence
            except Exception:
                pass

        return self._fallback_effect_label(effect_labels), 0.5

    def _canonicalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Project one row onto the canonical schema and keep only safe canonical fields."""

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

    def _build_frame(self, rows: list[dict[str, Any]]) -> Any:
        """Return a dataframe-like object with canonical and annotation columns."""

        columns = list(STANDARD_COLUMNS) + list(ANNOTATION_COLUMNS)
        try:
            import pandas as pd  # type: ignore[import-not-found]

            return pd.DataFrame(rows, columns=columns or None)
        except Exception:
            return _SimpleFrame(rows, columns)

    def _render_spec(self, spec: AnnotationSpec) -> str:
        """Render a stable markdown specification for human annotators."""

        sentiment_labels = spec.labels[:3]
        effect_labels = spec.labels[3:]
        sentiment_descriptions = [
            "low rating or clearly negative review",
            "mixed or middle rating review",
            "clearly positive review",
        ]
        sentiment_lines = [f"- {label}: {description}" for label, description in zip(sentiment_labels, sentiment_descriptions)]
        effect_lines = [f"- {label}: review mentions {label.replace('_', ' ')}" for label in effect_labels]
        example_effects = (effect_labels + [self._fallback_effect_label(effect_labels)])[:3]
        example_sentiments = ["positive", "neutral", "negative"]

        lines = [
            f"# Annotation Spec: {spec.name}",
            "",
            f"## Task\n{spec.metadata.get('task', 'annotation')}",
            "",
            "## Fields",
            f"- text: {spec.text_field}",
            f"- label: {spec.label_field}",
            f"- id: {spec.id_field}",
            "",
            "## Classes and definitions",
            "### Sentiment",
            *sentiment_lines,
            "",
            "### Effect",
            *effect_lines,
            "",
            "## Examples",
            *[f"- id={index + 1}, text=Example {index + 1}, rating={5 - index * 2} -> {sentiment} / {effect}" for index, (sentiment, effect) in enumerate(zip(example_sentiments, example_effects))],
            "",
            "## Edge cases",
            "- Missing rating -> sentiment_label is None",
            "- Empty text -> effect_label falls back to the configured default label",
            "- Invalid or out-of-vocabulary effect label -> configured default label",
        ]

        return "\n".join(lines)

    def _confidence_threshold(self) -> float:
        """Resolve the threshold used to flag low-confidence rows for HITL follow-up."""

        threshold = getattr(getattr(self.ctx, "config", None), "annotation", None)
        threshold_value = getattr(threshold, "confidence_threshold", 0.0) if threshold is not None else 0.0
        return float(threshold_value) if float(threshold_value) > 0 else 0.6

    def _coerce_confidence(self, value: Any) -> float:
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

    def _effect_label_vocabulary(self) -> list[str]:
        """Resolve the active effect-label vocabulary from config with a safe fitness fallback."""

        annotation = getattr(getattr(self.ctx, "config", None), "annotation", None)
        effect_labels = list(getattr(annotation, "effect_labels", []) or []) if annotation is not None else []
        return [str(label).strip() for label in effect_labels if str(label).strip()] or list(EFFECT_LABELS)

    def _fallback_effect_label(self, effect_labels: list[str]) -> str:
        """Choose the safest fallback effect label for offline and error paths."""

        if "other" in effect_labels:
            return "other"
        return effect_labels[0] if effect_labels else "other"

    def _normalize_label(self, value: Any) -> str:
        """Normalize a label for deterministic comparisons and fallback handling."""

        label = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        return label or "other"

    def _slugify(self, text: str) -> str:
        """Create a stable identifier for the spec name."""

        return "_".join(part for part in self._normalize_label(text).split("_") if part)


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