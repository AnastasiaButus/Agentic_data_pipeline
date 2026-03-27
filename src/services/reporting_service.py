"""Reporting helpers for the end-to-end demo pipeline."""

from __future__ import annotations

from pathlib import Path
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

    def write_eda_report(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> str:
        """Write a compact Russian EDA report for the post-quality dataset."""

        summary = self._build_extended_eda_summary(
            df_like,
            raw_df_like=raw_df_like,
            quality_report=quality_report,
        )
        lines = [
            "# EDA-пакет по данным после quality",
            "",
            "Это расширенный честный EDA-отчет по данным, которые реально идут дальше в pipeline.",
            "Он показывает структуру, сравнение raw/cleaned, пропуски и распределения, не подменяя пустые поля выдуманной статистикой.",
            "",
            f"- n_rows: {summary['n_rows']}",
            f"- column_count: {summary['column_count']}",
            f"- columns: {', '.join(summary['columns']) if summary['columns'] else 'нет'}",
            "",
            "## Raw vs cleaned",
        ]

        raw_vs_cleaned = summary["raw_vs_cleaned"]
        if raw_vs_cleaned["available"]:
            lines.extend([
                f"- raw_rows: {raw_vs_cleaned['raw_rows']}",
                f"- cleaned_rows: {raw_vs_cleaned['cleaned_rows']}",
                f"- dropped_rows: {raw_vs_cleaned['dropped_rows']}",
                f"- kept_fraction: {self._format_numeric(raw_vs_cleaned['kept_fraction'])}",
            ])
        else:
            lines.append(f"- {self._describe_absence(raw_vs_cleaned)}")

        lines.extend([
            "",
            "## Дубликаты",
        ])
        duplicate_summary = summary["duplicate_summary"]
        if duplicate_summary["available"]:
            lines.append(f"- duplicate_rows: {duplicate_summary['duplicate_rows']}")
        else:
            lines.append(f"- {self._describe_absence(duplicate_summary)}")

        lines.extend([
            "",
            "## Распределение source",
        ])

        source_distribution = summary["source_distribution"]
        if source_distribution["available"]:
            lines.append(f"- {self._format_count_map(source_distribution['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(source_distribution)}")

        lines.extend([
            "",
            "## Распределение effect_label",
        ])
        effect_label_distribution = summary["effect_label_distribution"]
        if effect_label_distribution["available"]:
            lines.append(f"- {self._format_count_map(effect_label_distribution['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(effect_label_distribution)}")

        lines.extend([
            "",
            "## Сводка rating",
        ])
        rating_summary = summary["rating_summary"]
        if rating_summary["available"]:
            lines.extend([
                f"- valid_count: {rating_summary['valid_count']}",
                f"- missing_or_invalid_count: {rating_summary['missing_or_invalid_count']}",
                f"- min: {self._format_numeric(rating_summary['min'])}",
                f"- max: {self._format_numeric(rating_summary['max'])}",
                f"- mean: {self._format_numeric(rating_summary['mean'])}",
            ])
        else:
            lines.append(f"- {self._describe_absence(rating_summary)}")

        lines.extend([
            "",
            "## Распределение rating",
        ])
        rating_distribution = summary["rating_distribution"]
        if rating_distribution["available"]:
            lines.append(f"- {self._format_count_map(rating_distribution['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(rating_distribution)}")

        lines.extend([
            "",
            "## Длина text",
        ])
        text_length_summary = summary["text_length_summary"]
        if text_length_summary["available"]:
            lines.extend([
                f"- valid_count: {text_length_summary['valid_count']}",
                f"- missing_or_invalid_count: {text_length_summary['missing_or_invalid_count']}",
                f"- min_chars: {self._format_numeric(text_length_summary['min_chars'])}",
                f"- max_chars: {self._format_numeric(text_length_summary['max_chars'])}",
                f"- mean_chars: {self._format_numeric(text_length_summary['mean_chars'])}",
            ])
        else:
            lines.append(f"- {self._describe_absence(text_length_summary)}")

        lines.extend([
            "",
            "## Бакеты длины text",
        ])
        text_length_buckets = summary["text_length_buckets"]
        if text_length_buckets["available"]:
            lines.append(f"- {self._format_count_map(text_length_buckets['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(text_length_buckets)}")

        lines.extend([
            "",
            "## Пропуски по ключевым колонкам",
        ])
        missing_values_summary = summary["missing_values_summary"]
        if missing_values_summary:
            for column_name, column_summary in missing_values_summary.items():
                if column_summary["available"]:
                    lines.append(
                        f"- {column_name}: missing_count={column_summary['missing_count']} / {summary['n_rows']}"
                    )
                else:
                    lines.append(f"- {column_name}: {self._describe_absence(column_summary)}")
        else:
            lines.append("- Ключевые колонки не найдены.")

        quality_warnings = summary["quality_warnings"]
        if quality_warnings:
            lines.extend(["", "## Quality notes"])
            for warning in quality_warnings:
                lines.append(f"- {warning}")

        if summary["notes"]:
            lines.extend(["", "## Гипотезы и примечания"])
            for note in summary["notes"]:
                lines.append(f"- {note}")

        path = Path("reports/eda_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_eda_context(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> str:
        """Write the machine-readable helper artifact for the EDA pack."""

        summary = self._build_extended_eda_summary(
            df_like,
            raw_df_like=raw_df_like,
            quality_report=quality_report,
        )
        path = Path("data/interim/eda_context.json")
        self.registry.save_json(path, summary)
        return str(path)

    def write_eda_html_report(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> str:
        """Write a self-contained offline HTML EDA report."""

        summary = self._build_extended_eda_summary(
            df_like,
            raw_df_like=raw_df_like,
            quality_report=quality_report,
        )
        cards = [
            ("Rows", str(summary["n_rows"])),
            ("Columns", str(summary["column_count"])),
            ("Duplicate rows", str(summary["duplicate_summary"].get("duplicate_rows", 0))),
            ("Warnings", str(len(summary["quality_warnings"]))),
        ]
        cards_html = "".join(
            f'<div class="card"><div class="label">{self._escape_html(label)}</div><div class="value">{self._escape_html(value)}</div></div>'
            for label, value in cards
        )
        sections = [
            self._html_metric_block("Columns", self._escape_html(", ".join(summary["columns"]) or "нет")),
            self._html_metric_block("Raw vs cleaned", self._format_raw_vs_cleaned(summary["raw_vs_cleaned"])),
            self._html_metric_block("Source distribution", self._format_distribution_html(summary["source_distribution"])),
            self._html_metric_block("Effect label distribution", self._format_distribution_html(summary["effect_label_distribution"])),
            self._html_metric_block("Rating summary", self._format_summary_dict_html(summary["rating_summary"])),
            self._html_metric_block("Rating distribution", self._format_distribution_html(summary["rating_distribution"])),
            self._html_metric_block("Text length summary", self._format_summary_dict_html(summary["text_length_summary"])),
            self._html_metric_block("Text length buckets", self._format_distribution_html(summary["text_length_buckets"])),
            self._html_metric_block("Missing values", self._format_missing_values_html(summary["missing_values_summary"])),
        ]
        if summary["quality_warnings"]:
            sections.append(
                self._html_metric_block(
                    "Quality notes",
                    "<ul>" + "".join(f"<li>{self._escape_html(note)}</li>" for note in summary["quality_warnings"]) + "</ul>",
                )
            )
        if summary["notes"]:
            sections.append(
                self._html_metric_block(
                    "Hypotheses",
                    "<ul>" + "".join(f"<li>{self._escape_html(note)}</li>" for note in summary["notes"]) + "</ul>",
                )
            )

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>EDA Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f1ea; color: #1f2933; margin: 0; padding: 32px; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .intro {{ max-width: 860px; line-height: 1.6; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 24px 0; }}
    .card {{ background: #fffaf5; border: 1px solid #d9c8b4; border-radius: 16px; padding: 18px; box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06); }}
    .label {{ font-size: 13px; color: #7a5c3e; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 30px; font-weight: 700; margin-top: 8px; }}
    .section-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .section {{ background: white; border-radius: 18px; padding: 20px; border: 1px solid #eadfce; }}
    .section h3 {{ margin: 0 0 12px 0; }}
    .plot {{ margin-top: 28px; background: white; border-radius: 18px; padding: 20px; border: 1px solid #eadfce; }}
    ul {{ margin: 0; padding-left: 20px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ text-align: left; padding: 6px 0; vertical-align: top; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>EDA Report</h1>
    <p class="intro">Offline-friendly HTML-отчёт по данным после quality stage. Он показывает структуру датасета, сравнение raw/cleaned, пропуски, распределения и краткие аналитические выводы.</p>
    <div class="cards">{cards_html}</div>
    <div class="section-grid">{''.join(sections)}</div>
    <div class="plot">
      <h2>Charts</h2>
      {self._build_eda_plotly_html(summary)}
    </div>
  </div>
</body>
</html>"""
        path = Path("reports/eda_report.html")
        self.registry.save_text(path, html)
        return str(path)

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

    def write_annotation_trace_report(self, annotation_trace: dict[str, Any]) -> str:
        """Write a Russian trace report for the annotation prompt and parser contract."""

        prompt_contract = annotation_trace.get("prompt_contract", {}) if isinstance(annotation_trace, dict) else {}
        parser_contract = annotation_trace.get("parser_contract", {}) if isinstance(annotation_trace, dict) else {}
        fallback_rows = annotation_trace.get("fallback_rows", []) if isinstance(annotation_trace, dict) else []

        lines = [
            "# Трассировка annotation contract",
            "",
            "Это компактный отчет о prompt contract, ожидаемом формате ответа и fallback-парсинге.",
            "Он нужен, чтобы будущий real LLM можно было подключить без угадывания контракта.",
            "",
            f"- llm_mode: {annotation_trace.get('llm_mode', 'unknown') if isinstance(annotation_trace, dict) else 'unknown'}",
            f"- n_rows: {annotation_trace.get('n_rows', 0) if isinstance(annotation_trace, dict) else 0}",
            f"- n_fallback_rows: {annotation_trace.get('n_fallback_rows', 0) if isinstance(annotation_trace, dict) else 0}",
            "",
            "## Prompt contract",
            f"- language: {prompt_contract.get('language', 'ru')}",
            f"- input_fields: {', '.join(prompt_contract.get('input_fields', [])) or 'нет'}",
            f"- output_fields: {', '.join(prompt_contract.get('output_fields', [])) or 'нет'}",
            f"- sentiment_labels: {', '.join(prompt_contract.get('sentiment_labels', [])) or 'нет'}",
            f"- effect_labels: {', '.join(prompt_contract.get('effect_labels', [])) or 'нет'}",
            "",
            "## Prompt preview",
            prompt_contract.get('prompt_preview', ''),
            "",
            "## Expected output",
            f"- example: {prompt_contract.get('expected_output_example', {})}",
            "",
            "## Parser contract",
            f"- preferred_format: {parser_contract.get('preferred_format', 'json')}",
            f"- accepted_fallbacks: {', '.join(parser_contract.get('accepted_fallbacks', [])) or 'нет'}",
            f"- parse_status_counts: {parser_contract.get('parse_status_counts', {})}",
            f"- fallback_reason_counts: {parser_contract.get('fallback_reason_counts', {})}",
        ]

        if fallback_rows:
            lines.extend(["", "## Fallback samples"])
            for row in fallback_rows:
                lines.append(
                    "- mode: {mode} | status: {status} | reasons: {reasons} | raw_output: {raw_output}".format(
                        mode=self._normalize_text(row.get("mode")),
                        status=self._normalize_text(row.get("parse_status")),
                        reasons=", ".join(row.get("fallback_reasons", []) or []) or "нет",
                        raw_output=self._normalize_text(row.get("raw_output")),
                    )
                )

        path = Path("reports/annotation_trace_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_annotation_trace_context(self, annotation_trace: dict[str, Any]) -> str:
        """Write a machine-readable trace artifact for the annotation prompt contract."""

        path = Path("data/interim/annotation_trace.json")
        self.registry.save_json(path, annotation_trace)
        return str(path)

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

    def write_review_queue_report(
        self,
        review_queue: Any,
        confidence_threshold: float,
        label_options: list[str],
    ) -> str:
        """Write a Russian review-pack report for human annotation review."""

        rows = self._to_records(review_queue)
        input_queue_path = "data/interim/review_queue.csv"
        corrected_queue_path = "data/interim/review_queue_corrected.csv"

        lines = [
            "# Очередь ручной проверки",
            "",
            "Это очередь примеров для ручной проверки после авторазметки.",
            "",
            "## Текущий этап",
            "",
            "- Этап pipeline: human review / HITL",
            "- Цель: проверить low-confidence примеры до retrain и финального обучения",
            "",
            "## Reviewer guide",
            "",
            f"- Входной файл очереди: {input_queue_path}",
            f"- Порог confidence: {self._format_numeric(confidence_threshold)}",
            f"- Строк в очереди: {len(rows)}",
            f"- Исправленный файл положите сюда: {corrected_queue_path}",
            "- Проверьте поля: id, source, text, label, effect_label, confidence, reviewed_effect_label, review_comment, human_verified",
            f"- Допустимые effect labels: {', '.join(label_options) if label_options else 'не заданы'}",
            "",
        ]

        if not rows:
            lines.extend(
                [
                    "Очередь пуста, ручная проверка не требуется.",
                    "",
                    "## Next step",
                    "",
                    "- Следующий шаг: active learning / training могут использовать текущий reviewed dataset без ручных правок.",
                ]
            )
        else:
            lines.extend(
                [
                    "## To-do reviewer",
                    "",
                    "1. Откройте `data/interim/review_queue.csv`.",
                    "2. Для спорных строк заполните `reviewed_effect_label`.",
                    "3. При необходимости добавьте `review_comment` и выставьте `human_verified=true`.",
                    "4. Сохраните исправленный файл как `data/interim/review_queue_corrected.csv`.",
                    "5. Перезапустите pipeline, чтобы merge применил ручные правки.",
                    "",
                ]
            )
            lines.append("## Примеры для проверки")
            lines.append("")
            for row in rows:
                lines.append(
                    "- id: {id} | source: {source} | effect_label: {effect_label} | confidence: {confidence} | text: {text}".format(
                        id=self._normalize_text(row.get("id")),
                        source=self._normalize_text(row.get("source")),
                        effect_label=self._normalize_text(row.get("effect_label")),
                        confidence=self._format_numeric(row.get("confidence")),
                        text=self._normalize_text(row.get("text")),
                    )
                )
            lines.extend(
                [
                    "",
                    "## Next step",
                    "",
                    "- Следующий шаг: загрузить corrected queue и повторно запустить pipeline для merge -> retrain.",
                ]
            )

        path = Path("reports/review_queue_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_review_queue_context(
        self,
        review_queue: Any,
        confidence_threshold: float,
        label_options: list[str],
    ) -> str:
        """Write a machine-readable helper artifact for review tooling."""

        rows = self._to_records(review_queue)
        payload = {
            "confidence_threshold": confidence_threshold,
            "n_rows": len(rows),
            "label_options": list(label_options),
            "input_queue_path": "data/interim/review_queue.csv",
            "expected_corrected_queue_path": "data/interim/review_queue_corrected.csv",
            "review_required": bool(rows),
            "current_stage": "human_review",
            "next_step": "fill_corrected_queue_and_rerun" if rows else "continue_to_active_learning_and_training",
            "review_columns": [
                "id",
                "source",
                "text",
                "label",
                "effect_label",
                "confidence",
                "reviewed_effect_label",
                "review_comment",
                "human_verified",
            ],
        }
        path = Path("data/interim/review_queue_context.json")
        self.registry.save_json(path, payload)
        return str(path)

    def write_review_merge_report(
        self,
        corrected_queue_found: bool,
        corrected_queue_path: str,
        n_corrected_rows: int,
        n_rows_with_reviewed_effect_label: int,
        n_effect_label_changes: int,
        reviewed_effect_labels: list[str],
        review_status: str,
    ) -> str:
        """Write a Russian markdown report for corrected-queue merge results."""

        lines = [
            "# Результат ручного merge",
            "",
            "Это краткий отчет о том, был ли найден corrected queue и что реально изменилось после ручной правки.",
            "",
            f"- corrected_queue_found: {'да' if corrected_queue_found else 'нет'}",
            f"- corrected_queue_path: {corrected_queue_path}",
            f"- review_status: {review_status}",
        ]

        if not corrected_queue_found:
            lines.extend([
                "",
                "Merge не выполнен, потому что corrected queue отсутствует.",
                "",
                "## Next step",
                "",
                "- Если нужна ручная валидация, заполните corrected queue и перезапустите pipeline.",
            ])
        else:
            lines.extend([
                f"- n_corrected_rows: {n_corrected_rows}",
                f"- n_rows_with_reviewed_effect_label: {n_rows_with_reviewed_effect_label}",
                f"- n_effect_label_changes: {n_effect_label_changes}",
                f"- reviewed_effect_labels: {', '.join(reviewed_effect_labels) if reviewed_effect_labels else 'нет'}",
            ])
            if review_status == "merged":
                lines.extend([
                    "",
                    "## Next step",
                    "",
                    "- Ручные правки применены. Следующий шаг: retrain / active learning на reviewed dataset.",
                ])
            else:
                lines.extend([
                    "",
                    "## Next step",
                    "",
                    "- Corrected queue обработан, но effect labels не изменились. Можно продолжать training на текущем датасете.",
                ])

        path = Path("reports/review_merge_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_review_merge_context(
        self,
        corrected_queue_found: bool,
        corrected_queue_path: str,
        n_corrected_rows: int,
        n_rows_with_reviewed_effect_label: int,
        n_effect_label_changes: int,
        reviewed_effect_labels: list[str],
        review_status: str,
    ) -> str:
        """Write a machine-readable helper artifact for review merge tooling."""

        payload = {
            "corrected_queue_found": corrected_queue_found,
            "corrected_queue_path": corrected_queue_path,
            "n_corrected_rows": n_corrected_rows,
            "n_rows_with_reviewed_effect_label": n_rows_with_reviewed_effect_label,
            "n_effect_label_changes": n_effect_label_changes,
            "reviewed_effect_labels": list(reviewed_effect_labels),
            "review_status": review_status,
        }
        path = Path("data/interim/review_merge_context.json")
        self.registry.save_json(path, payload)
        return str(path)

    def write_final_report(self, summary: dict[str, Any]) -> str:
        """Write the final end-to-end markdown report for the demo pipeline."""

        lines = ["# Final Report", ""]
        section_titles = {
            "runtime": "Runtime",
            "sources": "Sources",
            "quality": "Quality",
            "eda": "EDA",
            "annotation": "Annotation",
            "review": "Review",
            "approval": "Approval",
            "active_learning": "Active Learning",
            "training": "Training",
            "artifacts": "Artifacts",
        }
        for section_name in ["runtime", "sources", "quality", "eda", "annotation", "review", "approval", "active_learning", "training", "artifacts"]:
            section = summary.get(section_name)
            lines.append(f"## {section_titles[section_name]}")
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

    def _build_eda_summary(self, df_like: Any) -> dict[str, Any]:
        """Summarize the post-quality dataframe-like input without inventing values."""

        rows = self._to_records(df_like)
        columns = self._collect_columns(rows)
        if not columns:
            columns = self._collect_input_columns(df_like)
        columns_for_summary = columns or self._collect_input_columns(df_like)
        notes: list[str] = []

        if not rows:
            notes.append("Датасет пустой, поэтому статистика ограничена структурой входа.")

        source_distribution = self._build_distribution_summary(rows, "source", columns_for_summary)
        effect_label_distribution = self._build_distribution_summary(rows, "effect_label", columns_for_summary)
        rating_summary = self._build_numeric_summary(rows, "rating", columns_for_summary)
        text_length_summary = self._build_text_length_summary(rows, "text", columns_for_summary)
        missing_values_summary = self._build_missing_values_summary(
            rows,
            ["source", "effect_label", "rating", "text"],
            columns_for_summary,
        )

        if not columns:
            notes.append("Колонки не были переданы в dataframe-like input или не удалось извлечь записи.")

        return {
            "n_rows": len(rows),
            "columns": columns,
            "source_distribution": source_distribution,
            "effect_label_distribution": effect_label_distribution,
            "rating_summary": rating_summary,
            "text_length_summary": text_length_summary,
            "missing_values_summary": missing_values_summary,
            "notes": notes,
        }

    def _build_extended_eda_summary(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> dict[str, Any]:
        """Build a richer EDA payload for markdown, HTML, and helper artifacts."""

        summary = self._build_eda_summary(df_like)
        rows = self._to_records(df_like)
        columns = summary.get("columns", []) or self._collect_input_columns(df_like)
        raw_rows = self._to_records(raw_df_like) if raw_df_like is not None else []

        summary["column_count"] = len(columns)
        summary["duplicate_summary"] = self._build_duplicate_summary(rows, columns)
        summary["raw_vs_cleaned"] = self._build_raw_vs_cleaned_summary(raw_rows, rows)
        summary["rating_distribution"] = self._build_rating_distribution(rows, "rating", columns)
        summary["text_length_buckets"] = self._build_text_length_buckets(rows, "text", columns)
        summary["quality_warnings"] = self._extract_quality_warnings(quality_report)

        notes = list(summary.get("notes", []))
        if summary["duplicate_summary"].get("available") and summary["duplicate_summary"].get("duplicate_rows", 0) > 0:
            notes.append("В cleaned датасете ещё есть повторяющиеся строки, это стоит проверить перед финальным обучением.")

        raw_vs_cleaned = summary["raw_vs_cleaned"]
        if raw_vs_cleaned.get("available") and raw_vs_cleaned.get("dropped_rows", 0) > 0:
            notes.append(
                "После quality stage часть строк была удалена или отфильтрована. Это полезно сравнить с логикой чистки и approval flow."
            )

        rating_distribution = summary["rating_distribution"]
        if rating_distribution.get("available") and len(rating_distribution.get("counts", {})) <= 2:
            notes.append("Распределение rating выглядит узким. Для анализа выбросов и дисбаланса стоит смотреть не только на среднее.")

        summary["notes"] = notes
        return summary

    def _build_duplicate_summary(
        self,
        rows: list[dict[str, Any]],
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate duplicate rows directly from the cleaned dataframe-like input."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if not columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_values"}

        seen: set[str] = set()
        duplicate_rows = 0
        for row in rows:
            fingerprint = repr({column: row.get(column) for column in columns})
            if fingerprint in seen:
                duplicate_rows += 1
                continue
            seen.add(fingerprint)

        return {"available": True, "duplicate_rows": duplicate_rows}

    def _build_raw_vs_cleaned_summary(
        self,
        raw_rows: list[dict[str, Any]],
        cleaned_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare raw and cleaned row counts when the raw frame is available."""

        if not raw_rows:
            return {"available": False, "reason": "no_values"}

        raw_count = len(raw_rows)
        cleaned_count = len(cleaned_rows)
        return {
            "available": True,
            "raw_rows": raw_count,
            "cleaned_rows": cleaned_count,
            "dropped_rows": max(0, raw_count - cleaned_count),
            "kept_fraction": (cleaned_count / raw_count) if raw_count else 0.0,
        }

    def _build_rating_distribution(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a compact distribution for rating values."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        counts: dict[str, int] = {}
        for row in rows:
            value = row.get(column_name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value != numeric_value:
                continue

            bucket = self._format_numeric(numeric_value)
            counts[bucket] = counts.get(bucket, 0) + 1

        if not counts:
            return {"available": False, "reason": "no_numeric_values"}

        return {"available": True, "counts": counts}

    def _build_text_length_buckets(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build coarse text-length buckets for a quick EDA view."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        counts = {"0-49": 0, "50-99": 0, "100-199": 0, "200+": 0}
        has_values = False
        for row in rows:
            value = row.get(column_name)
            if self._is_missing_value(value):
                continue
            length = len(self._normalize_text(value))
            has_values = True
            if length < 50:
                counts["0-49"] += 1
            elif length < 100:
                counts["50-99"] += 1
            elif length < 200:
                counts["100-199"] += 1
            else:
                counts["200+"] += 1

        if not has_values:
            return {"available": False, "reason": "no_text_values"}

        return {"available": True, "counts": counts}

    def _extract_quality_warnings(self, quality_report: Any) -> list[str]:
        """Extract warning text from the optional quality-report payload."""

        if quality_report is None:
            return []

        if hasattr(quality_report, "as_dict"):
            payload = quality_report.as_dict()
        elif isinstance(quality_report, dict):
            payload = quality_report
        else:
            return []

        warnings = payload.get("warnings", [])
        if not isinstance(warnings, list):
            return []
        return [self._normalize_text(item) for item in warnings if self._normalize_text(item)]

    def _collect_columns(self, rows: list[dict[str, Any]]) -> list[str]:
        """Collect columns in first-seen order from row dictionaries."""

        columns: list[str] = []
        for row in rows:
            for key in row.keys():
                normalized_key = self._normalize_text(key)
                if normalized_key and normalized_key not in columns:
                    columns.append(normalized_key)
        return columns

    def _collect_input_columns(self, df_like: Any) -> list[str]:
        """Collect column names from dataframe-like inputs when row materialization is empty."""

        raw_columns = getattr(df_like, "columns", None)
        if raw_columns is None:
            return []

        columns: list[str] = []
        for column_name in list(raw_columns):
            normalized_column = self._normalize_text(column_name)
            if normalized_column and normalized_column not in columns:
                columns.append(normalized_column)
        return columns

    def _build_distribution_summary(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Count categorical values only when the requested column is present."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_values"}

        counts: dict[str, int] = {}
        for row in rows:
            value = self._normalize_text(row.get(column_name))
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1

        if not counts:
            return {"available": False, "reason": "no_values"}

        return {"available": True, "column": column_name, "counts": counts}

    def _build_numeric_summary(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize numeric values without clamping or synthetic defaults."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_numeric_values"}

        values: list[float] = []
        for row in rows:
            value = row.get(column_name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value != numeric_value:
                continue
            values.append(numeric_value)

        if not values:
            return {"available": False, "reason": "no_numeric_values"}

        return {
            "available": True,
            "column": column_name,
            "valid_count": len(values),
            "missing_or_invalid_count": len(rows) - len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def _build_text_length_summary(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize text length in characters for the existing text column."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_text_values"}

        lengths: list[int] = []
        for row in rows:
            value = row.get(column_name)
            if self._is_missing_value(value):
                continue

            normalized_value = self._normalize_text(value)

            lengths.append(len(normalized_value))

        if not lengths:
            return {"available": False, "reason": "no_text_values"}

        return {
            "available": True,
            "column": column_name,
            "valid_count": len(lengths),
            "missing_or_invalid_count": len(rows) - len(lengths),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "mean_chars": sum(lengths) / len(lengths),
        }

    def _build_missing_values_summary(
        self,
        rows: list[dict[str, Any]],
        key_columns: list[str],
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize missing values for the columns the report cares about most."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        summary: dict[str, Any] = {}
        for column_name in key_columns:
            if column_name not in columns:
                summary[column_name] = {"available": False, "reason": "column_absent"}
                continue

            missing_count = 0
            for row in rows:
                if self._is_missing_value(row.get(column_name)):
                    missing_count += 1

            summary[column_name] = {
                "available": True,
                "column": column_name,
                "missing_count": missing_count,
                "missing_fraction": missing_count / len(rows) if rows else 0.0,
            }

        return summary

    def _is_missing_value(self, value: Any) -> bool:
        """Detect missing textual values in a dataframe-friendly way."""

        if value is None:
            return True

        try:
            if value != value:
                return True
        except Exception:
            return True

        return not self._normalize_text(value)

    def _format_count_map(self, counts: dict[str, int]) -> str:
        """Render a short value-count list for markdown reports."""

        if not counts:
            return "нет значений"
        return ", ".join(f"{key}: {value}" for key, value in counts.items())

    def _describe_absence(self, payload: dict[str, Any]) -> str:
        """Explain why a metric is unavailable in a concise Russian phrase."""

        reason = payload.get("reason", "unknown")
        if reason == "column_absent":
            return "колонка отсутствует"
        if reason == "no_values":
            return "колонка есть, но значений нет"
        if reason == "no_numeric_values":
            return "числовые значения не найдены"
        if reason == "no_text_values":
            return "текстовые значения не найдены"
        return f"недоступно: {reason}"

    def _html_metric_block(self, title: str, body: str) -> str:
        """Render one HTML metric section for the EDA dashboard."""

        return f'<section class="section"><h3>{self._escape_html(title)}</h3>{body}</section>'

    def _escape_html(self, value: Any) -> str:
        """Escape text for safe inline HTML rendering."""

        text = self._normalize_text(value)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _format_distribution_html(self, payload: dict[str, Any]) -> str:
        """Render a small HTML table or an absence description for distributions."""

        if not payload.get("available"):
            return f'<p class="muted">{self._escape_html(self._describe_absence(payload))}</p>'

        counts = payload.get("counts", {})
        rows = "".join(
            f"<tr><td>{self._escape_html(key)}</td><td>{self._escape_html(value)}</td></tr>"
            for key, value in counts.items()
        )
        return f"<table><tbody>{rows}</tbody></table>"

    def _format_summary_dict_html(self, payload: dict[str, Any]) -> str:
        """Render a compact HTML view for summary dictionaries."""

        if not payload.get("available"):
            return f'<p class="muted">{self._escape_html(self._describe_absence(payload))}</p>'

        rows = "".join(
            f"<tr><td>{self._escape_html(key)}</td><td>{self._escape_html(self._format_numeric(value) if isinstance(value, (int, float)) else value)}</td></tr>"
            for key, value in payload.items()
            if key not in {"available", "column"}
        )
        return f"<table><tbody>{rows}</tbody></table>"

    def _format_missing_values_html(self, payload: dict[str, Any]) -> str:
        """Render missing-value summaries for the HTML report."""

        if not payload:
            return '<p class="muted">Нет данных.</p>'

        rows: list[str] = []
        for column_name, column_summary in payload.items():
            if column_summary.get("available"):
                rows.append(
                    f"<tr><td>{self._escape_html(column_name)}</td><td>{self._escape_html(column_summary.get('missing_count'))}</td><td>{self._escape_html(self._format_numeric(column_summary.get('missing_fraction')))}</td></tr>"
                )
            else:
                rows.append(
                    f"<tr><td>{self._escape_html(column_name)}</td><td colspan=\"2\">{self._escape_html(self._describe_absence(column_summary))}</td></tr>"
                )

        return "<table><thead><tr><th>Column</th><th>Missing</th><th>Fraction</th></tr></thead><tbody>{rows}</tbody></table>".format(
            rows="".join(rows)
        )

    def _format_raw_vs_cleaned(self, payload: dict[str, Any]) -> str:
        """Render raw-vs-cleaned comparison for the HTML report."""

        if not payload.get("available"):
            return f'<p class="muted">{self._escape_html(self._describe_absence(payload))}</p>'

        rows = "".join(
            f"<tr><td>{self._escape_html(key)}</td><td>{self._escape_html(self._format_numeric(value) if isinstance(value, (int, float)) else value)}</td></tr>"
            for key, value in payload.items()
            if key != "available"
        )
        return f"<table><tbody>{rows}</tbody></table>"

    def _build_eda_plotly_html(self, summary: dict[str, Any]) -> str:
        """Render inline Plotly charts when available, otherwise fall back to static HTML."""

        try:
            import plotly.graph_objects as go  # type: ignore[import-not-found]
        except Exception:
            return "<p class=\"muted\">Plotly не установлен, поэтому HTML-отчет показывает summary без интерактивных графиков.</p>"

        figures: list[str] = []
        chart_specs = [
            ("Source distribution", summary.get("source_distribution", {})),
            ("Effect label distribution", summary.get("effect_label_distribution", {})),
            ("Rating distribution", summary.get("rating_distribution", {})),
            ("Text length buckets", summary.get("text_length_buckets", {})),
        ]

        for index, (title, payload) in enumerate(chart_specs):
            if not payload.get("available"):
                continue
            counts = payload.get("counts", {})
            figure = go.Figure(
                data=[
                    go.Bar(
                        x=list(counts.keys()),
                        y=list(counts.values()),
                        marker_color="#b7791f",
                    )
                ]
            )
            figure.update_layout(
                title=title,
                template="plotly_white",
                margin=dict(l=24, r=24, t=56, b=24),
                height=360,
            )
            figures.append(
                figure.to_html(
                    full_html=False,
                    include_plotlyjs="inline" if index == 0 else False,
                )
            )

        if not figures:
            return "<p class=\"muted\">Для текущего датасета недостаточно значений, чтобы построить графики.</p>"
        return "".join(figures)

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
