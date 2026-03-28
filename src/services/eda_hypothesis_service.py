"""LLM-assisted and heuristic EDA hypothesis generation."""

from __future__ import annotations

import json
from typing import Any

from src.core.context import PipelineContext


class EDAHypothesisService:
    """Generate optional EDA hypotheses from machine-readable EDA context."""

    def __init__(self, ctx: PipelineContext, llm_client: Any | None = None) -> None:
        """Bind the active context and an optional structured-output LLM client."""

        self.ctx = ctx
        self.llm_client = llm_client

    def build_summary(self, eda_context: dict[str, Any] | None) -> dict[str, Any]:
        """Build a small operator-facing hypothesis summary from EDA context."""

        context = eda_context if isinstance(eda_context, dict) else {}
        use_llm_requested = bool(getattr(getattr(self.ctx.config, "annotation", None), "use_llm", False))
        requested_provider = self._normalize_provider(getattr(getattr(self.ctx.config, "annotation", None), "llm_provider", ""))
        resolved_provider = self._resolve_provider_name(self.llm_client)
        compact_context = self._compact_context(context)
        heuristic_hypotheses = self._build_heuristic_hypotheses(context)

        summary = {
            "available": bool(context),
            "hypothesis_mode": "heuristic_only",
            "use_llm_requested": use_llm_requested,
            "requested_provider": requested_provider,
            "resolved_provider": resolved_provider,
            "provider_status": "disabled" if not use_llm_requested else f"{resolved_provider}_available",
            "overall_note": "",
            "n_hypotheses": 0,
            "hypotheses": [],
            "hitl_followups": [],
            "compact_context": compact_context,
            "notes": [],
        }

        if not context:
            summary.update(
                {
                    "hypothesis_mode": "not_available_no_eda_context",
                    "provider_status": "not_available_no_eda_context",
                    "overall_note": "EDA context is missing, so no graph-grounded hypotheses can be proposed for this run.",
                    "notes": [
                        "Run EDA generation first, then reopen the dashboard or EDA hypothesis report.",
                    ],
                }
            )
            return summary

        hypotheses = heuristic_hypotheses
        overall_note = self._default_overall_note(context, hypotheses)
        notes = [
            "Hypotheses are suggestions for a human operator, not automatic pipeline decisions.",
        ]
        hypothesis_mode = "llm_disabled_heuristic_only"
        provider_status = "disabled_in_config" if not use_llm_requested else "heuristic_only"

        if use_llm_requested and resolved_provider == "gemini" and hasattr(self.llm_client, "generate_with_schema"):
            prompt = self._build_prompt(compact_context)
            try:
                raw_output = self.llm_client.generate_with_schema(prompt, self._response_schema())
                parsed = self._parse_llm_payload(raw_output)
                llm_hypotheses = parsed.get("hypotheses", [])
                if llm_hypotheses:
                    hypotheses = llm_hypotheses
                    overall_note = parsed.get("overall_note") or overall_note
                    hypothesis_mode = "llm_generate_parse"
                    provider_status = "gemini_active_for_eda_hypotheses"
                    notes.append("Gemini generated the current EDA hypotheses from eda_context.json, not from raw images.")
                else:
                    hypothesis_mode = "llm_empty_heuristic_fallback"
                    provider_status = "gemini_empty_heuristic_fallback"
                    notes.append("Gemini returned no usable EDA hypotheses, so the dashboard fell back to deterministic heuristics.")
            except Exception as exc:
                hypothesis_mode = "llm_error_heuristic_fallback"
                provider_status = "gemini_error_heuristic_fallback"
                notes.append(f"Gemini EDA hypothesis generation failed and fell back to deterministic heuristics: {self._normalize_text(exc)}")
        elif use_llm_requested and requested_provider == "gemini":
            hypothesis_mode = "gemini_requested_but_unavailable_heuristic_fallback"
            provider_status = "gemini_requested_but_unavailable_heuristic_fallback"
            notes.append("Gemini was requested for EDA hypotheses, but no active Gemini structured-output path was available in this run.")
        elif use_llm_requested:
            hypothesis_mode = "provider_not_supported_heuristic_fallback"
            provider_status = f"{resolved_provider}_not_supported_for_eda_hypotheses"
            notes.append("The current annotation provider does not support the structured EDA-hypothesis path, so deterministic heuristics stayed active.")

        summary.update(
            {
                "hypothesis_mode": hypothesis_mode,
                "provider_status": provider_status,
                "overall_note": overall_note,
                "n_hypotheses": len(hypotheses),
                "hypotheses": hypotheses,
                "hitl_followups": self._unique_followups(hypotheses),
                "notes": notes,
            }
        )
        return summary

    def _build_prompt(self, compact_context: dict[str, Any]) -> str:
        """Build a strict JSON-only prompt for graph-grounded EDA hypotheses."""

        payload = json.dumps(compact_context, ensure_ascii=False, indent=2)
        return "\n".join(
            [
                "Ты аналитик EDA и human-in-the-loop pipeline.",
                "На основе JSON summary ниже предложи до 4 коротких гипотез.",
                "Не выдумывай факты вне входного summary.",
                "Не принимай автоматических решений за пользователя.",
                "Каждая гипотеза должна быть связана с наблюдаемым сигналом и содержать действие для человека.",
                "Верни только JSON без markdown и пояснений.",
                "",
                "Ожидаемый JSON:",
                '{',
                '  "overall_note": "..." ,',
                '  "hypotheses": [',
                '    {',
                '      "title": "...",',
                '      "observation": "...",',
                '      "hypothesis": "...",',
                '      "hitl_action": "...",',
                '      "priority": "high" ',
                "    }",
                "  ]",
                "}",
                "",
                "Входной EDA summary:",
                payload,
            ]
        )

    def _response_schema(self) -> dict[str, Any]:
        """Return the structured-output schema for EDA hypotheses."""

        return {
            "type": "object",
            "additionalProperties": False,
            "propertyOrdering": ["overall_note", "hypotheses"],
            "properties": {
                "overall_note": {
                    "type": "string",
                    "description": "A short global note about the current EDA signals.",
                },
                "hypotheses": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "propertyOrdering": ["title", "observation", "hypothesis", "hitl_action", "priority"],
                        "properties": {
                            "title": {"type": "string"},
                            "observation": {"type": "string"},
                            "hypothesis": {"type": "string"},
                            "hitl_action": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["title", "observation", "hypothesis", "hitl_action", "priority"],
                    },
                },
            },
            "required": ["overall_note", "hypotheses"],
        }

    def _parse_llm_payload(self, raw_output: Any) -> dict[str, Any]:
        """Parse and sanitize a structured LLM hypothesis response."""

        raw_text = self._normalize_text(raw_output)
        if not raw_text:
            return {"overall_note": "", "hypotheses": []}

        candidates = [raw_text]
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            fragment = raw_text[start : end + 1]
            if fragment not in candidates:
                candidates.append(fragment)

        payload: dict[str, Any] = {}
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break

        if not payload:
            return {"overall_note": "", "hypotheses": []}

        hypotheses_payload = payload.get("hypotheses", []) if isinstance(payload.get("hypotheses"), list) else []
        hypotheses: list[dict[str, str]] = []
        for item in hypotheses_payload[:4]:
            if not isinstance(item, dict):
                continue
            title = self._normalize_text(item.get("title"))
            observation = self._normalize_text(item.get("observation"))
            hypothesis = self._normalize_text(item.get("hypothesis"))
            hitl_action = self._normalize_text(item.get("hitl_action"))
            priority = self._normalize_priority(item.get("priority"))
            if not (title and observation and hypothesis and hitl_action):
                continue
            hypotheses.append(
                {
                    "title": title,
                    "observation": observation,
                    "hypothesis": hypothesis,
                    "hitl_action": hitl_action,
                    "priority": priority,
                }
            )

        return {
            "overall_note": self._normalize_text(payload.get("overall_note")),
            "hypotheses": hypotheses,
        }

    def _build_heuristic_hypotheses(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """Build deterministic fallback hypotheses directly from EDA signals."""

        hypotheses: list[dict[str, str]] = []
        duplicate_summary = context.get("duplicate_summary", {}) if isinstance(context.get("duplicate_summary"), dict) else {}
        if duplicate_summary.get("available") and int(duplicate_summary.get("duplicate_rows", 0) or 0) > 0:
            duplicate_rows = int(duplicate_summary.get("duplicate_rows", 0) or 0)
            hypotheses.append(
                {
                    "title": "Duplicate rows remain after cleaning",
                    "observation": f"The cleaned dataset still contains {duplicate_rows} duplicate row(s).",
                    "hypothesis": "Collection or schema normalization may be preserving near-identical records that should be deduplicated before retrain.",
                    "hitl_action": "Inspect duplicate examples and confirm whether deduplication rules should be tightened before the next rerun.",
                    "priority": "high",
                }
            )

        raw_vs_cleaned = context.get("raw_vs_cleaned", {}) if isinstance(context.get("raw_vs_cleaned"), dict) else {}
        if raw_vs_cleaned.get("available") and int(raw_vs_cleaned.get("dropped_rows", 0) or 0) > 0:
            dropped_rows = int(raw_vs_cleaned.get("dropped_rows", 0) or 0)
            hypotheses.append(
                {
                    "title": "Quality stage is removing a visible share of rows",
                    "observation": f"Raw-vs-cleaned comparison reports {dropped_rows} dropped row(s).",
                    "hypothesis": "The current cleaning policy is probably filtering noisy, empty, or too-short texts aggressively enough to change the effective training subset.",
                    "hitl_action": "Compare raw and cleaned examples, then decide whether the quality filters match the intended task before rerun.",
                    "priority": "high" if dropped_rows > 2 else "medium",
                }
            )

        missing_values_summary = (
            context.get("missing_values_summary", {}) if isinstance(context.get("missing_values_summary"), dict) else {}
        )
        missing_columns = []
        for column_name, payload in missing_values_summary.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("available") and int(payload.get("missing_count", 0) or 0) > 0:
                missing_columns.append(
                    f"{column_name}: {int(payload.get('missing_count', 0) or 0)} missing"
                )
        if missing_columns:
            hypotheses.append(
                {
                    "title": "Some key fields are still partially missing",
                    "observation": "; ".join(missing_columns[:3]),
                    "hypothesis": "The collection or normalization path may be uneven across sources, leaving fields incomplete for a subset of rows.",
                    "hitl_action": "Review raw/source-specific records and decide whether collection rules or approval scope should be adjusted before retraining.",
                    "priority": "high" if any(item.startswith("text:") for item in missing_columns) else "medium",
                }
            )

        label_distribution = (
            context.get("effect_label_distribution", {}) if isinstance(context.get("effect_label_distribution"), dict) else {}
        )
        label_counts = label_distribution.get("counts", {}) if isinstance(label_distribution.get("counts"), dict) else {}
        if label_distribution.get("available") and label_counts:
            total = sum(int(value or 0) for value in label_counts.values()) or 1
            dominant_label, dominant_count = max(label_counts.items(), key=lambda item: int(item[1] or 0))
            dominant_fraction = int(dominant_count or 0) / total
            if dominant_fraction >= 0.65:
                hypotheses.append(
                    {
                        "title": "One effect label currently dominates the cleaned batch",
                        "observation": f"{self._normalize_text(dominant_label)} accounts for {dominant_fraction:.0%} of effect_label values.",
                        "hypothesis": "The current source mix or task framing may be producing a narrow supervised signal that can bias training and review priorities.",
                        "hitl_action": "Inspect source approval and class balance before the next retrain; consider whether a broader source subset is needed.",
                        "priority": "medium",
                    }
                )

        word_cloud = context.get("cleaned_word_cloud", {}) if isinstance(context.get("cleaned_word_cloud"), dict) else {}
        if word_cloud.get("available"):
            terms = word_cloud.get("terms", []) if isinstance(word_cloud.get("terms"), list) else []
            if terms:
                top_terms = [self._normalize_text(item.get("term")) for item in terms[:4] if isinstance(item, dict)]
                hypotheses.append(
                    {
                        "title": "The cleaned vocabulary suggests a narrow topical focus",
                        "observation": f"Top cleaned terms: {', '.join(term for term in top_terms if term) or 'n/a'}.",
                        "hypothesis": "The dataset appears coherent around a limited topic cluster, which is good for demo stability but may underrepresent edge cases.",
                        "hitl_action": "Use the EDA report and review queue to decide whether the current topic focus is acceptable or if discovery should be widened.",
                        "priority": "low",
                    }
                )

        quality_warnings = context.get("quality_warnings", []) if isinstance(context.get("quality_warnings"), list) else []
        if quality_warnings:
            hypotheses.append(
                {
                    "title": "Quality warnings point to operator review points",
                    "observation": self._normalize_text(quality_warnings[0]),
                    "hypothesis": "The current run already surfaced a data-quality condition that should be validated by a human before the next major retrain decision.",
                    "hitl_action": "Inspect quality notes and confirm whether cleaning strategy, approval gate, or review threshold should be changed.",
                    "priority": "medium",
                }
            )

        if not hypotheses:
            hypotheses.append(
                {
                    "title": "The cleaned dataset looks stable enough for a controlled demo run",
                    "observation": "EDA did not surface strong duplicate, missing-value, or imbalance signals that require immediate intervention.",
                    "hypothesis": "The current batch is suitable for continuing through HITL and retrain, provided the operator still performs a quick manual spot-check.",
                    "hitl_action": "Proceed to review workspace and final reports, but keep a manual spot-check before presenting the run as final.",
                    "priority": "low",
                }
            )

        return hypotheses[:4]

    def _compact_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Keep only the most useful EDA signals for a small hypothesis prompt."""

        if not context:
            return {}

        word_cloud = context.get("cleaned_word_cloud", {}) if isinstance(context.get("cleaned_word_cloud"), dict) else {}
        top_terms = word_cloud.get("terms", []) if isinstance(word_cloud.get("terms"), list) else []
        return {
            "n_rows": context.get("n_rows"),
            "column_count": context.get("column_count"),
            "columns": context.get("columns", []),
            "raw_vs_cleaned": context.get("raw_vs_cleaned", {}),
            "duplicate_summary": context.get("duplicate_summary", {}),
            "source_distribution": context.get("source_distribution", {}),
            "effect_label_distribution": context.get("effect_label_distribution", {}),
            "rating_summary": context.get("rating_summary", {}),
            "rating_distribution": context.get("rating_distribution", {}),
            "text_length_summary": context.get("text_length_summary", {}),
            "text_length_buckets": context.get("text_length_buckets", {}),
            "missing_values_summary": context.get("missing_values_summary", {}),
            "quality_warnings": context.get("quality_warnings", []),
            "notes": context.get("notes", []),
            "cleaned_word_cloud": {
                "available": bool(word_cloud.get("available")),
                "valid_text_rows": word_cloud.get("valid_text_rows"),
                "token_count": word_cloud.get("token_count"),
                "unique_terms": word_cloud.get("unique_terms"),
                "top_terms": top_terms[:8],
            },
        }

    def _default_overall_note(self, context: dict[str, Any], hypotheses: list[dict[str, str]]) -> str:
        """Build a safe default overall note when LLM output is unavailable."""

        row_count = int(context.get("n_rows", 0) or 0)
        dropped_rows = 0
        raw_vs_cleaned = context.get("raw_vs_cleaned", {}) if isinstance(context.get("raw_vs_cleaned"), dict) else {}
        if raw_vs_cleaned.get("available"):
            dropped_rows = int(raw_vs_cleaned.get("dropped_rows", 0) or 0)
        if hypotheses and row_count:
            return (
                f"The current EDA context covers {row_count} cleaned row(s). "
                f"Hypotheses below are advisory and should be validated by a human before changing quality, approval, or retrain settings."
            )
        if dropped_rows:
            return "EDA remained mostly deterministic, but row drops and quality filters still deserve a human check before the next rerun."
        return "EDA hypotheses were generated from machine-readable summary signals and should be treated as operator guidance, not automatic decisions."

    def _unique_followups(self, hypotheses: list[dict[str, str]]) -> list[str]:
        """Deduplicate HITL follow-up actions while preserving order."""

        seen: set[str] = set()
        followups: list[str] = []
        for item in hypotheses:
            action = self._normalize_text(item.get("hitl_action"))
            if not action or action in seen:
                continue
            seen.add(action)
            followups.append(action)
        return followups

    def _resolve_provider_name(self, llm_client: Any) -> str:
        """Map the active client to a stable provider label."""

        if llm_client is None:
            return "disabled"
        provider_name = llm_client.__class__.__name__.strip().lower()
        if provider_name == "geminiclient":
            return "gemini"
        if provider_name == "mockllm":
            return "mock"
        return provider_name or "unknown"

    def _normalize_priority(self, value: Any) -> str:
        """Normalize priority values into a small stable vocabulary."""

        normalized = self._normalize_text(value).lower()
        if normalized in {"high", "medium", "low"}:
            return normalized
        return "medium"

    def _normalize_provider(self, value: Any) -> str:
        """Normalize provider names while keeping blanks explicit."""

        normalized = self._normalize_text(value).lower().replace(" ", "_").replace("-", "_")
        return normalized or "disabled"

    def _normalize_text(self, value: Any) -> str:
        """Normalize arbitrary values into stable strings."""

        if value is None:
            return ""
        return str(value).strip()
