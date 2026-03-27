"""Reporting helpers for the end-to-end demo pipeline."""

from __future__ import annotations

from typing import Any

from src.core.context import PipelineContext
from src.services.artifact_registry import ArtifactRegistry


class ReportingService:
    """Render markdown reports for each pipeline stage and the final summary."""

    def __init__(self, ctx: PipelineContext, registry: ArtifactRegistry | None = None) -> None:
        """Bind the reporting service to the active context and artifact registry."""

        self.ctx = ctx
        self.registry = registry if registry is not None else ArtifactRegistry(ctx)

    def write_source_report(self, sources: list[Any]) -> str:
        """Write a compact Russian shortlist report for manual approval review."""

        approval_candidates = [self._candidate_to_approval_record(candidate) for candidate in sources]
        self.registry.save_json("data/raw/approval_candidates.json", approval_candidates)

        lines = [
            "# Короткий shortlist источников",
            "",
            "Это список найденных источников для ручного просмотра и одобрения перед следующим шагом pipeline.",
            "",
        ]

        if not sources:
            lines.extend([
                "Кандидаты не найдены.",
                "",
                "Чтобы одобрить источники, добавьте их `source_id` в `data/raw/approved_sources.json`.",
                "Формат файла: JSON list of strings.",
            ])
            path = "reports/source_report.md"
            self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
            return path

        lines.extend([
            "Чтобы одобрить источники, добавьте их `source_id` в `data/raw/approved_sources.json`.",
            "Формат файла: JSON list of strings.",
            "",
        ])

        for index, candidate in enumerate(sources, start=1):
            title = self._normalize_text(getattr(candidate, "title", ""))
            source_id = self._normalize_text(getattr(candidate, "source_id", ""))
            source_type = self._normalize_text(getattr(candidate, "source_type", ""))
            uri = self._normalize_text(getattr(candidate, "uri", ""))
            score = self._format_numeric(getattr(candidate, "score", 0.0))
            metadata = getattr(candidate, "metadata", None)

            lines.append(f"## Источник {index}")
            lines.append(f"- source_id: {source_id}")
            lines.append(f"- source_type: {source_type}")
            lines.append(f"- title: {title}")
            lines.append(f"- uri: {uri}")
            lines.append(f"- score: {score}")

            metadata_text = self._format_compact_metadata(metadata)
            if metadata_text:
                lines.append(f"- metadata: {metadata_text}")
            lines.append("")

        path = "reports/source_report.md"
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return path

    def write_quality_report(self, quality_report: Any) -> str:
        """Write a quality summary that is easy to drop into README-style docs."""

        payload = quality_report.as_dict() if hasattr(quality_report, "as_dict") else dict(quality_report)
        lines = ["# Quality Report", ""]
        lines.append(f"- missing: {payload.get('missing', {})}")
        lines.append(f"- duplicates: {payload.get('duplicates', 0)}")
        lines.append(f"- outliers: {payload.get('outliers', {})}")
        lines.append(f"- imbalance: {payload.get('imbalance', {})}")
        lines.append(f"- warnings: {payload.get('warnings', [])}")
        path = "reports/quality_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_annotation_report(self, df_labeled: Any, annotation_summary: dict[str, Any] | None = None) -> str:
        """Write a monitoring report that emphasizes effect labels and confidence."""

        rows = self._to_records(df_labeled)
        effect_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        low_confidence = 0
        threshold = 0.7
        if annotation_summary and annotation_summary.get("confidence_threshold") is not None:
            threshold = float(annotation_summary["confidence_threshold"])

        for row in rows:
            effect_label = self._normalize_text(row.get("effect_label")) or "other"
            effect_counts[effect_label] = effect_counts.get(effect_label, 0) + 1
            confidence = self._coerce_float(row.get("confidence"))
            confidence_values.append(confidence)
            if confidence < threshold:
                low_confidence += 1

        confidence_mean = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        lines = ["# Annotation Report", "", f"- n_rows: {len(rows)}", f"- effect_label_distribution: {effect_counts}", f"- confidence_mean: {confidence_mean:.3f}", f"- low_confidence: {low_confidence}"]
        path = "reports/annotation_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_al_report(self, history: list[dict[str, Any]]) -> str:
        """Write an active-learning report with the per-iteration history."""

        lines = ["# Active Learning Report", ""]
        for row in history:
            lines.append(
                f"- iteration {row.get('iteration')}: n_labeled={row.get('n_labeled')} accuracy={row.get('accuracy'):.3f} f1={row.get('f1'):.3f}"
            )
        path = "reports/al_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_al_comparison_report(self, rows: list[dict[str, Any]]) -> str:
        """Write a compact markdown summary for AL strategy comparison rows.

        The report stays table-based so it remains offline-safe and easy to inspect in plain text.
        """

        lines = ["# Active Learning Comparison Report", ""]
        lines.append("| strategy | iteration | n_labeled | accuracy | f1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                "| {strategy} | {iteration} | {n_labeled} | {accuracy:.3f} | {f1:.3f} |".format(
                    strategy=row.get("strategy", ""),
                    iteration=row.get("iteration", ""),
                    n_labeled=row.get("n_labeled", ""),
                    accuracy=self._coerce_float(row.get("accuracy")),
                    f1=self._coerce_float(row.get("f1")),
                )
            )

        path = "reports/al_comparison_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_final_report(self, summary: dict[str, Any]) -> str:
        """Write the final end-to-end markdown report for the demo pipeline."""

        lines = ["# Final Report", ""]
        for section_name in ["sources", "quality", "annotation", "review", "approval", "active_learning", "training", "artifacts"]:
            section = summary.get(section_name)
            lines.append(f"## {section_name.replace('_', ' ').title()}")
            lines.append("")
            if isinstance(section, dict):
                for key, value in section.items():
                    lines.append(f"- {key}: {value}")
            elif isinstance(section, list):
                for item in section:
                    lines.append(f"- {item}")
            else:
                lines.append(f"- {section}")
            lines.append("")

        path = "final_report.md"
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return path

    def _to_records(self, df: Any) -> list[dict[str, Any]]:
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

    def _normalize_text(self, value: Any) -> str:
        """Normalize arbitrary values into stable strings for reporting."""

        if value is None:
            return ""
        return str(value).strip()

    def _coerce_float(self, value: Any) -> float:
        """Convert a value to float while tolerating missing confidences."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric != numeric:
            return 0.0
        return max(0.0, min(1.0, numeric))

    def _format_numeric(self, value: Any) -> str:
        """Format numeric discovery values without confidence-style clamping."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return self._normalize_text(value)

        if numeric != numeric:
            return "nan"

        if numeric.is_integer():
            return str(int(numeric))

        return f"{numeric:.3f}".rstrip("0").rstrip(".")

    def _format_compact_metadata(self, metadata: Any) -> str:
        """Render a short metadata summary that stays readable in markdown reports."""

        if not isinstance(metadata, dict) or not metadata:
            return ""

        preferred_keys = ["web_url", "downloads", "likes", "tags", "stars", "language"]
        parts: list[str] = []
        seen_keys: set[str] = set()

        for key in preferred_keys:
            if key in metadata:
                parts.append(f"{key}={metadata[key]}")
                seen_keys.add(key)

        for key, value in metadata.items():
            if key in seen_keys:
                continue
            if len(parts) >= 8:
                break
            parts.append(f"{key}={value}")

        return ", ".join(parts)

    def _candidate_to_approval_record(self, candidate: Any) -> dict[str, Any]:
        """Convert a shortlist candidate into a stable helper artifact row.

        The helper artifact is intentionally simple so a human can inspect the markdown report
        while an approval workflow can read the JSON shortlist without extra parsing logic.
        """

        metadata = getattr(candidate, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "source_id": self._normalize_text(getattr(candidate, "source_id", "")),
            "source_type": self._normalize_text(getattr(candidate, "source_type", "")),
            "title": self._normalize_text(getattr(candidate, "title", "")),
            "uri": self._normalize_text(getattr(candidate, "uri", "")),
            "score": getattr(candidate, "score", 0.0),
            "metadata": dict(metadata),
        }