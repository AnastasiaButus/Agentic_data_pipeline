"""Annotation stage agent for deterministic spec generation, auto-labeling, and export prep."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.constants import STANDARD_COLUMNS
from src.domain import AnnotationSpec
from src.providers.labelstudio.exporter import to_labelstudio_tasks
from src.providers.labelstudio.validators import validate_labelstudio_tasks
from src.providers.llm.mock_llm import MockLLM
from src.services.artifact_registry import ArtifactRegistry

try:  # pragma: no cover - exercised when sklearn is installed in the runtime.
    from sklearn.metrics import cohen_kappa_score
except ImportError:  # pragma: no cover - fallback keeps local/offline tests runnable.
    cohen_kappa_score = None  # type: ignore[assignment]


ANNOTATION_COLUMNS = ("sentiment_label", "effect_label", "confidence")
EFFECT_LABELS = ["energy", "side_effects", "other"]
ANNOTATION_OUTPUT_FIELDS = ["effect_label", "sentiment_label", "confidence"]
ANNOTATION_SENTIMENT_LABELS = ["negative", "neutral", "positive"]


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
        self._last_annotation_trace: dict[str, Any] = {}

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
        trace_rows: list[dict[str, Any]] = []

        for row in rows:
            canonical_row = self._canonicalize_row(row)
            fallback_sentiment = self.map_rating_to_sentiment(canonical_row.get("rating"))
            annotation_result = self._classify_annotation_row(canonical_row, effect_labels, fallback_sentiment)
            canonical_row.update(
                {
                    "sentiment_label": annotation_result["sentiment_label"],
                    "effect_label": annotation_result["effect_label"],
                    "confidence": annotation_result["confidence"],
                }
            )
            output_rows.append(canonical_row)
            trace_rows.append(annotation_result)

        self._last_annotation_trace = self._build_annotation_trace(rows, output_rows, effect_labels, trace_rows)

        return self._build_frame(output_rows)

    def build_annotation_prompt(self, text: Any, effect_labels: list[str] | None = None) -> str:
        """Build the Russian prompt contract for future LLM-based annotation.

        The prompt is intentionally narrow: it only asks for effect_label, sentiment_label,
        and confidence so the contract stays easy to parse and stable for offline tests.
        """

        labels = effect_labels if effect_labels is not None else self._effect_label_vocabulary()
        labels = [self._normalize_label(label) for label in labels if self._normalize_label(label)]
        if not labels:
            labels = list(EFFECT_LABELS)

        text_value = self._normalize_optional_text(text) or "<empty>"
        return "\n".join(
            [
                "Ты разметчик отзывов о пищевых добавках.",
                "Верни только JSON без пояснений, markdown и лишнего текста.",
                "",
                "Заполни ровно эти поля:",
                "- effect_label: одно из значений списка ниже",
                "- sentiment_label: negative, neutral или positive",
                "- confidence: число от 0 до 1",
                "",
                f"Допустимые effect_label: {', '.join(labels)}",
                "",
                "Правила:",
                "- Не добавляй новые поля.",
                "- Если текст неполный или неоднозначный, выбирай самый безопасный вариант.",
                "- confidence должен отражать уверенность модели, но оставаться в диапазоне 0..1.",
                "",
                "Ожидаемый ответ:",
                '{"effect_label": "...", "sentiment_label": "...", "confidence": 0.0}',
                "",
                "Текст отзыва:",
                text_value,
            ]
        )

    def parse_annotation_output(
        self,
        output: Any,
        effect_labels: list[str] | None = None,
        *,
        fallback_sentiment: str | None = None,
    ) -> dict[str, Any]:
        """Parse an LLM annotation response and apply safe fallbacks for partial output.

        The parser accepts JSON first, then simple key-value lines, and finally falls back to
        deterministic defaults without failing the pipeline.
        """

        labels = effect_labels if effect_labels is not None else self._effect_label_vocabulary()
        labels = [self._normalize_label(label) for label in labels if self._normalize_label(label)] or list(EFFECT_LABELS)
        fallback_effect = self._fallback_effect_label(labels)
        fallback_sentiment = self._normalize_sentiment(fallback_sentiment)
        raw_text = self._normalize_optional_text(output)

        parsed_payload = self._extract_annotation_payload(raw_text)
        parse_status = "fallback"
        fallback_reasons: list[str] = []

        effect_label = fallback_effect
        sentiment_label = fallback_sentiment
        confidence = 0.5

        if parsed_payload:
            parse_status = "partial_fallback"

            parsed_effect_label = self._normalize_optional_label(parsed_payload.get("effect_label"))
            if parsed_effect_label and parsed_effect_label in labels:
                effect_label = parsed_effect_label
            else:
                fallback_reasons.append("effect_label")

            parsed_sentiment_label = self._normalize_sentiment(parsed_payload.get("sentiment_label"))
            if parsed_sentiment_label is not None:
                sentiment_label = parsed_sentiment_label
            elif fallback_sentiment is not None:
                fallback_reasons.append("sentiment_label")

            parsed_confidence = self._parse_confidence(parsed_payload.get("confidence"))
            if parsed_confidence is not None:
                confidence = parsed_confidence
            else:
                fallback_reasons.append("confidence")

            if not fallback_reasons:
                parse_status = "parsed"
            elif len(fallback_reasons) == 3:
                parse_status = "fallback"

        else:
            fallback_reasons.extend(["effect_label", "sentiment_label", "confidence"])

        return {
            "effect_label": effect_label,
            "sentiment_label": sentiment_label,
            "confidence": self._coerce_confidence(confidence),
            "parse_status": parse_status,
            "fallback_reasons": fallback_reasons,
            "raw_output": raw_text,
        }

    def get_annotation_trace(self) -> dict[str, Any]:
        """Return the most recent annotation trace for reporting and tests."""

        return deepcopy(self._last_annotation_trace)

    def check_quality(self, df_labeled: Any) -> dict[str, Any]:
        """Summarize the labeled batch for future HITL routing and confidence filtering."""

        rows = self._to_records(df_labeled)
        if not rows:
            return {
                "label_dist": {},
                "confidence_mean": 0.0,
                "n_low_confidence": 0,
                "n_rows": 0,
                "confidence_threshold": self._confidence_threshold(),
                "agreement": None,
                "kappa": None,
            }

        label_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        threshold = self._confidence_threshold()
        n_low_confidence = 0
        effect_labels = self._effect_label_vocabulary()
        fallback_label = self._fallback_effect_label(effect_labels)
        use_effect_labels = any(self._normalize_optional_label(row.get("effect_label")) for row in rows)
        auto_labels: list[str] = []
        human_labels: list[str] = []

        for row in rows:
            raw_label = row.get("effect_label") if use_effect_labels else row.get("label")
            label = self._normalize_optional_label(raw_label) or fallback_label
            label_counts[label] = label_counts.get(label, 0) + 1

            reviewed_label = self._normalize_optional_label(row.get("reviewed_effect_label"))
            auto_effect_label = self._normalize_optional_label(row.get("effect_label"))
            if reviewed_label and auto_effect_label:
                auto_labels.append(auto_effect_label)
                human_labels.append(reviewed_label)

            confidence = self._coerce_confidence(row.get("confidence"))
            confidence_values.append(confidence)
            if confidence < threshold:
                n_low_confidence += 1

        total = sum(label_counts.values())
        label_dist = {label: count / total for label, count in label_counts.items()}
        confidence_mean = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        agreement: float | None = None
        kappa: float | None = None
        if human_labels:
            matches = sum(1 for auto_label, human_label in zip(auto_labels, human_labels) if auto_label == human_label)
            agreement = matches / len(human_labels)
            if cohen_kappa_score is not None:
                try:
                    kappa = float(cohen_kappa_score(human_labels, auto_labels))
                except Exception:
                    kappa = None

        return {
            "label_dist": label_dist,
            "confidence_mean": confidence_mean,
            "n_low_confidence": n_low_confidence,
            "n_rows": len(rows),
            "confidence_threshold": threshold,
            "agreement": agreement,
            "kappa": kappa,
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

    def _classify_annotation_row(
        self,
        row: dict[str, Any],
        effect_labels: list[str],
        fallback_sentiment: str | None,
    ) -> dict[str, Any]:
        """Classify one row while preserving a trace of the contract and fallback path."""

        text = row.get("text")
        if self.llm_client is not None and hasattr(self.llm_client, "classify_effect"):
            try:
                result = self.llm_client.classify_effect(str(text or ""), labels=effect_labels)
                effect_label = self._normalize_label(getattr(result, "label", "other"))
                if effect_label not in effect_labels:
                    effect_label = self._fallback_effect_label(effect_labels)
                confidence = self._coerce_confidence(getattr(result, "confidence", 0.5))
                return {
                    "effect_label": effect_label,
                    "sentiment_label": fallback_sentiment,
                    "confidence": confidence,
                    "mode": "classify_effect",
                    "parse_status": "direct",
                    "fallback_reasons": [],
                    "prompt": self.build_annotation_prompt(text, effect_labels),
                    "raw_output": getattr(result, "as_dict", lambda: {})(),
                }
            except Exception:
                return self._fallback_annotation_row(text, effect_labels, fallback_sentiment, mode="classify_effect_error")

        if self.llm_client is not None and hasattr(self.llm_client, "generate"):
            prompt = self.build_annotation_prompt(text, effect_labels)
            try:
                raw_output = self.llm_client.generate(prompt)
            except Exception:
                return self._fallback_annotation_row(text, effect_labels, fallback_sentiment, mode="generate_error", prompt=prompt)

            parsed = self.parse_annotation_output(raw_output, effect_labels, fallback_sentiment=fallback_sentiment)
            parsed.update({"mode": "generate_parse", "prompt": prompt})
            return parsed

        return self._fallback_annotation_row(text, effect_labels, fallback_sentiment, mode="offline_fallback")

    def _fallback_annotation_row(
        self,
        text: Any,
        effect_labels: list[str],
        fallback_sentiment: str | None,
        *,
        mode: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Return a safe deterministic annotation result when the LLM path is unavailable."""

        effect_label, confidence = self._predict_effect(text, effect_labels)
        return {
            "effect_label": effect_label,
            "sentiment_label": fallback_sentiment,
            "confidence": confidence,
            "mode": mode,
            "parse_status": "fallback",
            "fallback_reasons": ["effect_label", "sentiment_label", "confidence"],
            "prompt": prompt or self.build_annotation_prompt(text, effect_labels),
            "raw_output": "",
        }

    def _build_annotation_trace(
        self,
        input_rows: list[dict[str, Any]],
        output_rows: list[dict[str, Any]],
        effect_labels: list[str],
        row_traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarize prompt, parser, and fallback behavior for reporting.

        The trace stays compact so it can be written as a small JSON helper artifact.
        """

        fallback_count = sum(1 for row in row_traces if row.get("parse_status") in {"partial_fallback", "fallback"})
        parse_status_counts: dict[str, int] = {}
        fallback_reason_counts: dict[str, int] = {}
        for row in row_traces:
            status = self._normalize_text(row.get("parse_status")) or "unknown"
            parse_status_counts[status] = parse_status_counts.get(status, 0) + 1
            for reason in row.get("fallback_reasons", []) or []:
                reason_name = self._normalize_text(reason)
                if reason_name:
                    fallback_reason_counts[reason_name] = fallback_reason_counts.get(reason_name, 0) + 1

        prompt_preview = self.build_annotation_prompt(input_rows[0].get("text") if input_rows else "", effect_labels)
        expected_output_example = {
            "effect_label": effect_labels[0] if effect_labels else "other",
            "sentiment_label": "positive",
            "confidence": 0.5,
        }

        return {
            "prompt_contract": {
                "language": "ru",
                "task": "auto_annotation",
                "input_fields": ["text", "rating"],
                "output_fields": list(ANNOTATION_OUTPUT_FIELDS),
                "sentiment_labels": list(ANNOTATION_SENTIMENT_LABELS),
                "effect_labels": list(effect_labels),
                "prompt_preview": prompt_preview,
                "expected_output_example": expected_output_example,
            },
            "parser_contract": {
                "preferred_format": "json",
                "accepted_fallbacks": ["key_value", "partial_json", "deterministic_fallback"],
                "parse_status_counts": parse_status_counts,
                "fallback_reason_counts": fallback_reason_counts,
            },
            "llm_mode": self._resolve_annotation_mode(),
            "n_rows": len(output_rows),
            "n_fallback_rows": fallback_count,
            "fallback_rows": [row for row in row_traces if row.get("parse_status") in {"partial_fallback", "fallback"}][:5],
        }

    def _extract_annotation_payload(self, raw_text: str) -> dict[str, Any]:
        """Extract an annotation payload from JSON or key-value LLM output."""

        if not raw_text:
            return {}

        candidates = [raw_text]
        json_fragment = self._extract_json_fragment(raw_text)
        if json_fragment and json_fragment not in candidates:
            candidates.append(json_fragment)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload

        return self._parse_key_value_payload(raw_text)

    def _extract_json_fragment(self, raw_text: str) -> str:
        """Extract the first JSON object-like fragment from a raw LLM response."""

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return raw_text[start : end + 1]

    def _parse_key_value_payload(self, raw_text: str) -> dict[str, Any]:
        """Parse simple key-value annotation output when JSON is missing."""

        payload: dict[str, Any] = {}
        for line in raw_text.splitlines():
            match = re.match(r"^\s*([a-zA-Z_]+)\s*[:=]\s*(.+?)\s*$", line)
            if not match:
                continue
            key = self._normalize_text(match.group(1)).lower()
            value = match.group(2).strip().strip('"').strip("'")
            payload[key] = value
        return payload

    def _parse_confidence(self, value: Any) -> float | None:
        """Parse confidence without crashing on invalid or partially formatted values."""

        if value is None:
            return None
        try:
            numeric = float(str(value).strip().rstrip("%"))
        except (TypeError, ValueError):
            return None
        if numeric > 1.0 and numeric <= 100.0:
            numeric = numeric / 100.0
        if numeric != numeric:
            return None
        return self._coerce_confidence(numeric)

    def _normalize_sentiment(self, value: Any) -> str | None:
        """Normalize sentiment labels while keeping missing values explicit."""

        if value is None:
            return None
        normalized = self._normalize_text(value).lower().replace(" ", "_").replace("-", "_")
        if not normalized:
            return None
        if normalized in ANNOTATION_SENTIMENT_LABELS:
            return normalized
        return None

    def _resolve_annotation_mode(self) -> str:
        """Describe which annotation path is active for trace reporting."""

        if self.llm_client is None:
            return "offline_fallback"
        if hasattr(self.llm_client, "classify_effect"):
            return "classify_effect"
        if hasattr(self.llm_client, "generate"):
            return "generate_parse"
        return "offline_fallback"

    def _normalize_text(self, value: Any) -> str:
        """Normalize arbitrary values into stable strings for prompt and trace helpers."""

        if value is None:
            return ""
        return str(value).strip()

    def _normalize_optional_text(self, value: Any) -> str:
        """Normalize text while preserving empty content as an empty string."""

        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

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

    def _normalize_optional_label(self, value: Any) -> str:
        """Normalize a label while preserving missing values as empty strings."""

        if value is None:
            return ""
        label = str(value).strip()
        if not label:
            return ""
        return self._normalize_label(label)

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