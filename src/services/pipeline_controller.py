"""Top-level orchestration for the demo fitness supplements pipeline."""

from __future__ import annotations

from typing import Any
import os

from src.agents.active_learning_agent import ActiveLearningAgent
from src.agents.annotation_agent import AnnotationAgent
from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.data_quality_agent import DataQualityAgent
from src.core.context import PipelineContext
from src.core.runtime import build_runtime_summary
from src.providers.llm.gemini_client import GeminiClient
from src.providers.llm.mock_llm import MockLLM
from src.services.review_queue_service import CORRECTED_QUEUE_PATH, ReviewQueueService
from src.services.review_agreement import build_review_agreement_summary
from src.services.reporting_service import ReportingService
from src.services.source_governance import build_online_governance_summary
from src.services.source_discovery_service import SourceDiscoveryService
from src.services.training_service import TrainingService


class PipelineController:
    """Run discovery, collection, quality, annotation, review, AL, training, and reporting."""

    def __init__(
        self,
        ctx: PipelineContext,
        discovery_service: SourceDiscoveryService | None = None,
        collection_agent: DataCollectionAgent | None = None,
        quality_agent: DataQualityAgent | None = None,
        annotation_agent: AnnotationAgent | None = None,
        review_queue_service: ReviewQueueService | None = None,
        active_learning_agent: ActiveLearningAgent | None = None,
        training_service: TrainingService | None = None,
        reporting_service: ReportingService | None = None,
    ) -> None:
        """Bind the controller to the active context and optionally injected components."""

        self.ctx = ctx
        self.discovery_service = discovery_service if discovery_service is not None else SourceDiscoveryService(ctx)
        self.collection_agent = collection_agent if collection_agent is not None else DataCollectionAgent(ctx)
        self.quality_agent = quality_agent if quality_agent is not None else DataQualityAgent(ctx)
        if annotation_agent is not None:
            self.annotation_agent = annotation_agent
        else:
            llm_client = self._build_annotation_llm_client()
            self.annotation_agent = AnnotationAgent(ctx, llm_client=llm_client)
        self.review_queue_service = review_queue_service if review_queue_service is not None else ReviewQueueService(ctx)
        self.active_learning_agent = active_learning_agent if active_learning_agent is not None else ActiveLearningAgent(ctx)
        self.training_service = training_service if training_service is not None else TrainingService(ctx)
        self.reporting_service = reporting_service if reporting_service is not None else ReportingService(ctx)

    def _build_annotation_llm_client(self) -> Any | None:
        """Select the annotation provider explicitly from config while keeping the offline baseline stable."""

        annotation_config = getattr(self.ctx.config, "annotation", None)
        use_llm = bool(getattr(annotation_config, "use_llm", False))
        if not use_llm:
            return None

        provider = str(getattr(annotation_config, "llm_provider", "") or "").strip().lower()
        if provider in {"", "mock"}:
            return MockLLM()

        if provider == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                return GeminiClient(api_key=gemini_api_key)
            return MockLLM()

        return MockLLM()

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end and return a compact run summary."""

        runtime_summary = build_runtime_summary(self.ctx.config)
        sources = self.discovery_service.run()
        source_report_path = self.reporting_service.write_source_report(sources)
        online_governance_getter = getattr(self.discovery_service, "get_online_governance_summary", None)
        if callable(online_governance_getter):
            online_governance_detail = online_governance_getter(sources)
        else:
            online_governance_detail = build_online_governance_summary(self.ctx.config, sources)
        online_governance_report_path = self.reporting_service.write_online_governance_report(online_governance_detail)
        online_governance_context_path = self.reporting_service.write_online_governance_context(online_governance_detail)

        approval_file_exists = self.discovery_service.registry.exists("data/raw/approved_sources.json")
        approved_sources = self.discovery_service.load_approved_candidates(sources)
        if not approval_file_exists:
            approval_status = "skipped_missing_file"
        elif not approved_sources:
            approval_status = "applied_empty_subset"
        else:
            approval_status = "applied"

        collected = self.collection_agent.run(approved_sources)
        quality_report = self.quality_agent.detect_issues(collected)
        cleaned = self.quality_agent.run(collected)
        quality_report_path = self.reporting_service.write_quality_report(quality_report)
        eda_report_path = self.reporting_service.write_eda_report(
            cleaned,
            raw_df_like=collected,
            quality_report=quality_report,
        )
        eda_context_path = self.reporting_service.write_eda_context(
            cleaned,
            raw_df_like=collected,
            quality_report=quality_report,
        )
        eda_html_report_path = self.reporting_service.write_eda_html_report(
            cleaned,
            raw_df_like=collected,
            quality_report=quality_report,
        )

        annotated = self.annotation_agent.auto_label(cleaned)
        annotation_summary = self.annotation_agent.check_quality(annotated)
        annotation_trace_getter = getattr(self.annotation_agent, "get_annotation_trace", None)
        annotation_trace = annotation_trace_getter() if callable(annotation_trace_getter) else {}
        annotation_trace_report_path = self.reporting_service.write_annotation_trace_report(annotation_trace)
        annotation_trace_context_path = self.reporting_service.write_annotation_trace_context(annotation_trace)
        annotation_report_path = self.reporting_service.write_annotation_report(annotated, annotation_summary)

        # Export the reviewer queue before consuming corrected labels so HITL semantics stay consistent.
        review_threshold = self._resolve_review_threshold(annotation_summary)
        label_options = self._resolve_review_label_options()
        annotation_llm_summary = self._build_annotation_llm_summary(annotation_trace, label_options)
        review_queue = self.review_queue_service.export_low_confidence_queue(annotated, threshold=review_threshold)
        review_queue_report_path = self.reporting_service.write_review_queue_report(review_queue, review_threshold, label_options)
        review_queue_context_path = self.reporting_service.write_review_queue_context(review_queue, review_threshold, label_options)
        review_queue_rows = len(self._to_records(review_queue))
        review_required = review_queue_rows > 0

        reviewed = annotated
        review_status = "skipped_missing_corrected_queue"
        review_merge_report_path = ""
        review_merge_context_path = ""
        try:
            corrected_queue = self.review_queue_service.load_corrected_queue()
        except FileNotFoundError:
            corrected_queue = None

        if corrected_queue is not None:
            reviewed = self.review_queue_service.merge_reviewed_labels(annotated, corrected_queue)

        review_merge_summary = self._build_review_merge_summary(annotated, corrected_queue, review_status)
        
        # Adjust status based on actual effect_label changes
        if review_merge_summary["corrected_queue_found"]:
            n_changes = review_merge_summary["n_effect_label_changes"]
            review_status = "merged" if n_changes > 0 else "merged_no_changes"
            review_merge_summary["review_status"] = review_status
        review_merge_report_path = self.reporting_service.write_review_merge_report(
            review_merge_summary["corrected_queue_found"],
            review_merge_summary["corrected_queue_path"],
            review_merge_summary["n_corrected_rows"],
            review_merge_summary["n_rows_with_reviewed_effect_label"],
            review_merge_summary["n_effect_label_changes"],
            review_merge_summary["reviewed_effect_labels"],
            review_merge_summary["review_status"],
        )
        review_merge_context_path = self.reporting_service.write_review_merge_context(
            review_merge_summary["corrected_queue_found"],
            review_merge_summary["corrected_queue_path"],
            review_merge_summary["n_corrected_rows"],
            review_merge_summary["n_rows_with_reviewed_effect_label"],
            review_merge_summary["n_effect_label_changes"],
            review_merge_summary["reviewed_effect_labels"],
            review_merge_summary["review_status"],
        )
        review_agreement_summary = build_review_agreement_summary(annotated, corrected_queue)
        review_agreement_report_path = self.reporting_service.write_review_agreement_report(review_agreement_summary)
        review_agreement_context_path = self.reporting_service.write_review_agreement_context(review_agreement_summary)
        training_comparison_summary = self._build_training_comparison_summary(
            annotated,
            reviewed,
            review_status=review_status,
            review_required=review_required,
            corrected_queue_found=review_merge_summary["corrected_queue_found"],
            n_effect_label_changes=review_merge_summary["n_effect_label_changes"],
        )
        training_comparison_report_path = self.reporting_service.write_training_comparison_report(training_comparison_summary)
        training_comparison_context_path = self.reporting_service.write_training_comparison_context(training_comparison_summary)

        reviewed_rows = self._to_records(reviewed)
        al_seed_size = max(1, min(50, len(reviewed_rows) // 2 if len(reviewed_rows) > 1 else 1))
        al_batch_size = max(1, min(20, len(reviewed_rows) // 3 if len(reviewed_rows) > 2 else 1))
        al_iterations = 3 if len(reviewed_rows) < 10 else 5

        al_history, al_labeled = self.active_learning_agent.run_cycle(
            reviewed,
            strategy="entropy",
            seed_size=al_seed_size,
            n_iterations=al_iterations,
            batch_size=al_batch_size,
        )
        al_report_path = self.reporting_service.write_al_report(al_history)
        al_comparison_payload = self._build_active_learning_comparison_payload(
            reviewed,
            entropy_history=al_history,
            seed_size=al_seed_size,
            n_iterations=al_iterations,
            batch_size=al_batch_size,
        )
        al_comparison_report_path = self.reporting_service.write_al_comparison_report(al_comparison_payload)
        al_comparison_context_path = self.reporting_service.write_al_comparison_context(al_comparison_payload)

        artifacts, training_metrics = self.training_service.train(al_labeled)
        reviewer_action = (
            "fill data/interim/review_queue_corrected.csv and rerun pipeline"
            if review_required and corrected_queue is None
            else "review queue already processed or not required"
        )
        next_step = (
            "human review rerun recommended before final retrain"
            if review_required and corrected_queue is None
            else "active learning and training completed for current run"
        )
        dashboard_path = "reports/run_dashboard.html"
        dashboard_status = "attention_required" if review_required and corrected_queue is None else "completed"
        dashboard_stage = "human_review" if review_required and corrected_queue is None else "completed"
        final_report_path = "final_report.md"
        review_workspace_path = "reports/review_workspace.html"

        final_summary = {
            "runtime": runtime_summary,
            "dashboard": {
                "dashboard_path": dashboard_path,
                "final_report_path": final_report_path,
                "pipeline_status": dashboard_status,
                "current_stage": dashboard_stage,
                "primary_action": reviewer_action,
                "next_step": next_step,
            },
            "sources": {
                "n_candidates": len(sources),
                "source_report_path": source_report_path,
            },
            "online_governance": {
                "governance_report_path": online_governance_report_path,
                "governance_context_path": online_governance_context_path,
                "remote_sources_enabled": online_governance_detail.get("remote_sources_enabled"),
                "active_provider_count": online_governance_detail.get("active_provider_count"),
                "providers_requiring_attention": online_governance_detail.get("providers_requiring_attention"),
                "github_auth_mode": online_governance_detail.get("github_auth_mode"),
                "fallback_strategy": online_governance_detail.get("fallback_strategy"),
            },
            "quality": {
                "quality_report_path": quality_report_path,
                "warnings": getattr(quality_report, "warnings", []),
            },
            "eda": {
                "eda_report_path": eda_report_path,
                "eda_html_report_path": eda_html_report_path,
                "eda_context_path": eda_context_path,
                "n_rows": len(self._to_records(cleaned)),
            },
            "annotation": {
                "annotation_report_path": annotation_report_path,
                "annotation_trace_report_path": annotation_trace_report_path,
                "annotation_trace_context_path": annotation_trace_context_path,
                "confidence_threshold": annotation_summary.get("confidence_threshold"),
                "n_low_confidence": annotation_summary.get("n_low_confidence"),
                "use_llm_requested": annotation_llm_summary.get("use_llm_requested"),
                "requested_provider": annotation_llm_summary.get("requested_provider"),
                "resolved_provider": annotation_llm_summary.get("resolved_provider"),
                "provider_status": annotation_llm_summary.get("provider_status"),
                "llm_mode": annotation_llm_summary.get("llm_mode"),
                "n_fallback_rows": annotation_llm_summary.get("n_fallback_rows"),
                "effect_labels": annotation_llm_summary.get("effect_labels"),
                "fallback_reason_counts": annotation_llm_summary.get("fallback_reason_counts"),
            },
            "review": {
                "status": review_status,
                "review_queue_rows": review_queue_rows,
                "review_required": review_required,
                "reviewer_action": reviewer_action,
                "next_step": next_step,
                "review_workspace_path": review_workspace_path,
                "review_queue_report_path": review_queue_report_path,
                "review_queue_context_path": review_queue_context_path,
                "review_merge_report_path": review_merge_report_path,
                "review_merge_context_path": review_merge_context_path,
            },
            "agreement": {
                "agreement_report_path": review_agreement_report_path,
                "agreement_context_path": review_agreement_context_path,
                "comparison_scope": review_agreement_summary.get("comparison_scope"),
                "compared_rows": review_agreement_summary.get("compared_rows"),
                "agreement": review_agreement_summary.get("agreement"),
                "kappa": review_agreement_summary.get("kappa"),
                "kappa_status": review_agreement_summary.get("kappa_status"),
            },
            "approval": {
                "approved_sources_path": "data/raw/approved_sources.json",
                "n_approved_sources": len(approved_sources),
                "approval_status": approval_status,
            },
            "active_learning": {
                "al_report_path": al_report_path,
                "al_comparison_report_path": al_comparison_report_path,
                "al_comparison_context_path": al_comparison_context_path,
                "comparison_scope": al_comparison_payload.get("comparison_scope"),
                "best_strategy": al_comparison_payload.get("best_strategy"),
                "delta_f1_entropy_minus_random": al_comparison_payload.get("delta_f1_entropy_minus_random"),
                "delta_accuracy_entropy_minus_random": al_comparison_payload.get("delta_accuracy_entropy_minus_random"),
                "history": al_history,
            },
            "training_comparison": {
                "comparison_report_path": training_comparison_report_path,
                "comparison_context_path": training_comparison_context_path,
                "comparison_scope": training_comparison_summary.get("comparison_scope"),
                "baseline_status": training_comparison_summary.get("baseline_status"),
                "reviewed_status": training_comparison_summary.get("reviewed_status"),
                "datasets_identical": training_comparison_summary.get("datasets_identical"),
                "delta_accuracy": training_comparison_summary.get("delta_accuracy"),
                "delta_f1": training_comparison_summary.get("delta_f1"),
                "n_effect_label_changes": training_comparison_summary.get("n_effect_label_changes"),
            },
            "training": training_metrics,
            "artifacts": artifacts,
        }

        final_report_path = self.reporting_service.write_final_report(final_summary)
        review_workspace_path = self.reporting_service.write_review_workspace(
            review_queue,
            review_threshold,
            label_options,
            review_required=review_required,
            corrected_queue_found=review_merge_summary["corrected_queue_found"],
            corrected_queue_path=review_merge_summary["corrected_queue_path"],
            review_status=review_merge_summary["review_status"],
            next_step=next_step,
            dashboard_path=dashboard_path,
            final_report_path=final_report_path,
            review_queue_report_path=review_queue_report_path,
            review_queue_context_path=review_queue_context_path,
            review_merge_report_path=review_merge_report_path,
            review_merge_context_path=review_merge_context_path,
        )
        dashboard_path = self.reporting_service.write_run_dashboard(final_summary)
        review_workspace_path = self.reporting_service.write_review_workspace(
            review_queue,
            review_threshold,
            label_options,
            review_required=review_required,
            corrected_queue_found=review_merge_summary["corrected_queue_found"],
            corrected_queue_path=review_merge_summary["corrected_queue_path"],
            review_status=review_merge_summary["review_status"],
            next_step=next_step,
            dashboard_path=dashboard_path,
            final_report_path=final_report_path,
            review_queue_report_path=review_queue_report_path,
            review_queue_context_path=review_queue_context_path,
            review_merge_report_path=review_merge_report_path,
            review_merge_context_path=review_merge_context_path,
        )

        return {
            "sources": sources,
            "collected": collected,
            "cleaned": cleaned,
            "annotated": annotated,
            "reviewed": reviewed,
            "active_learning_history": al_history,
            "trained_artifacts": artifacts,
            "training_metrics": training_metrics,
            "reports": {
                "source_report": source_report_path,
                "online_governance_report": online_governance_report_path,
                "online_governance_context": online_governance_context_path,
                "quality_report": quality_report_path,
                "eda_report": eda_report_path,
                "eda_html_report": eda_html_report_path,
                "eda_context": eda_context_path,
                "annotation_report": annotation_report_path,
                "annotation_trace_report": annotation_trace_report_path,
                "annotation_trace_context": annotation_trace_context_path,
                "review_workspace": review_workspace_path,
                "review_queue_report": review_queue_report_path,
                "review_queue_context": review_queue_context_path,
                "review_merge_report": review_merge_report_path,
                "review_merge_context": review_merge_context_path,
                "review_agreement_report": review_agreement_report_path,
                "review_agreement_context": review_agreement_context_path,
                "al_report": al_report_path,
                "al_comparison_report": al_comparison_report_path,
                "al_comparison_context": al_comparison_context_path,
                "training_comparison_report": training_comparison_report_path,
                "training_comparison_context": training_comparison_context_path,
                "dashboard": dashboard_path,
                "final_report": final_report_path,
            },
            "review_status": review_status,
            "approved_sources": approved_sources,
            "approval_status": approval_status,
            "runtime_mode": runtime_summary["effective_mode"],
        }

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

    def _review_threshold(self) -> float:
        """Return the review queue threshold used for export and reporting.

        The demo configs define a confidence threshold, but the review queue export must keep a
        stable fallback when the configured value is missing or left at the dataclass default.
        """

        annotation = getattr(self.ctx.config, "annotation", None)
        configured_threshold = getattr(annotation, "confidence_threshold", None)
        if configured_threshold in (None, 0, 0.0):
            return 0.7
        return float(configured_threshold)

    def _resolve_review_threshold(self, annotation_summary: dict[str, Any] | None) -> float:
        """Prefer the annotation summary threshold and fall back to the review default.

        This keeps the review pack aligned with the actual annotation pass while preserving the
        existing fallback when a summary omits the threshold.
        """

        summary_threshold = annotation_summary.get("confidence_threshold") if annotation_summary else None
        if summary_threshold is not None:
            return float(summary_threshold)
        return self._review_threshold()

    def _resolve_review_label_options(self) -> list[str]:
        """Resolve the effect-label vocabulary used by the review pack.

        The controller follows the same vocabulary semantics as AnnotationAgent and only falls
        back to config values or the demo-safe default when the agent helper is unavailable.
        """

        effect_labels_helper = getattr(self.annotation_agent, "_effect_label_vocabulary", None)
        if callable(effect_labels_helper):
            label_options = list(effect_labels_helper())
        else:
            annotation = getattr(self.ctx.config, "annotation", None)
            label_options = list(getattr(annotation, "effect_labels", []) or []) if annotation is not None else []

        cleaned = [str(label).strip() for label in label_options if str(label).strip()]
        return cleaned or ["energy", "side_effects", "other"]

    def _build_annotation_llm_summary(
        self,
        annotation_trace: dict[str, Any],
        label_options: list[str],
    ) -> dict[str, Any]:
        """Summarize how annotation currently uses LLM-assisted versus fallback paths."""

        annotation_config = getattr(self.ctx.config, "annotation", None)
        use_llm_requested = bool(getattr(annotation_config, "use_llm", False))
        requested_provider = self._normalize_optional_provider(getattr(annotation_config, "llm_provider", ""))
        llm_client = getattr(self.annotation_agent, "llm_client", None)
        resolved_provider = self._resolve_annotation_provider_name(llm_client)

        if not use_llm_requested:
            provider_status = "disabled_in_config"
        elif requested_provider == "gemini" and resolved_provider == "gemini":
            provider_status = "gemini_active"
        elif requested_provider == "gemini" and resolved_provider == "mock":
            provider_status = "gemini_requested_but_mock_fallback_active"
        elif resolved_provider == "mock":
            provider_status = "offline_mock_llm_active"
        elif resolved_provider == "disabled":
            provider_status = "disabled"
        else:
            provider_status = f"{resolved_provider}_active"

        prompt_contract = (
            annotation_trace.get("prompt_contract", {})
            if isinstance(annotation_trace.get("prompt_contract"), dict)
            else {}
        )
        parser_contract = (
            annotation_trace.get("parser_contract", {})
            if isinstance(annotation_trace.get("parser_contract"), dict)
            else {}
        )
        effect_labels = prompt_contract.get("effect_labels", label_options)
        if not isinstance(effect_labels, list):
            effect_labels = list(label_options)
        fallback_reason_counts = parser_contract.get("fallback_reason_counts", {})
        if not isinstance(fallback_reason_counts, dict):
            fallback_reason_counts = {}

        return {
            "use_llm_requested": use_llm_requested,
            "requested_provider": requested_provider,
            "resolved_provider": resolved_provider,
            "provider_status": provider_status,
            "llm_mode": self._normalize_optional_provider(annotation_trace.get("llm_mode")),
            "n_fallback_rows": int(annotation_trace.get("n_fallback_rows", 0) or 0),
            "effect_labels": [str(label).strip() for label in effect_labels if str(label).strip()],
            "fallback_reason_counts": {str(key): int(value) for key, value in fallback_reason_counts.items()},
        }

    def _resolve_annotation_provider_name(self, llm_client: Any) -> str:
        """Map the active annotation client to a small stable provider name for reporting."""

        if llm_client is None:
            return "disabled"

        provider_name = llm_client.__class__.__name__.strip().lower()
        if provider_name == "mockllm":
            return "mock"
        if provider_name == "geminiclient":
            return "gemini"
        return provider_name or "unknown"

    def _normalize_optional_provider(self, value: Any) -> str:
        """Normalize provider-like values while keeping blanks explicit."""

        normalized = self._normalize_text(value).lower().replace(" ", "_").replace("-", "_")
        return normalized or "disabled"

    def _build_review_merge_summary(
        self,
        annotated: Any,
        corrected_queue: Any,
        review_status: str,
    ) -> dict[str, Any]:
        """Summarize the real merge outcome without changing merge semantics."""

        corrected_rows = self._to_records(corrected_queue) if corrected_queue is not None else []
        annotated_rows = self._to_records(annotated)
        annotated_by_id = {
            row_id: dict(row)
            for row in annotated_rows
            if (row_id := self._normalize_merge_id(row.get("id"))) is not None
        }

        reviewed_effect_labels: list[str] = []
        n_effect_label_changes = 0
        n_rows_with_reviewed_effect_label = 0
        n_valid_corrected_rows = 0

        for row in corrected_rows:
            row_id = self._normalize_merge_id(row.get("id"))
            if row_id is None:
                continue

            n_valid_corrected_rows += 1

            reviewed_effect_label = self._normalize_merge_label(row.get("reviewed_effect_label"))
            if reviewed_effect_label:
                n_rows_with_reviewed_effect_label += 1
                if reviewed_effect_label not in reviewed_effect_labels:
                    reviewed_effect_labels.append(reviewed_effect_label)

            original_effect_label = self._normalize_merge_label(annotated_by_id.get(row_id, {}).get("effect_label"))
            if reviewed_effect_label and reviewed_effect_label != original_effect_label:
                n_effect_label_changes += 1

        corrected_queue_found = corrected_queue is not None
        corrected_queue_path = CORRECTED_QUEUE_PATH

        return {
            "corrected_queue_found": corrected_queue_found,
            "corrected_queue_path": corrected_queue_path,
            "n_corrected_rows": n_valid_corrected_rows,
            "n_rows_with_reviewed_effect_label": n_rows_with_reviewed_effect_label,
            "n_effect_label_changes": n_effect_label_changes,
            "reviewed_effect_labels": reviewed_effect_labels,
            "review_status": review_status,
        }

    def _normalize_merge_id(self, value: Any) -> str | None:
        """Normalize merge ids and drop missing or blank values instead of stringifying them."""

        normalized = self._normalize_text(value)
        return normalized if normalized else None

    def _normalize_merge_label(self, value: Any) -> str:
        """Normalize merge labels for stable comparison in the review summary."""

        normalized = self._normalize_text(value)
        if not normalized:
            return ""
        return normalized.lower().replace(" ", "_").replace("-", "_")

    def _normalize_text(self, value: Any) -> str:
        """Normalize arbitrary values into stable strings for merge and reporting helpers."""

        if value is None:
            return ""
        return str(value).strip()

    def _build_training_comparison_summary(
        self,
        annotated: Any,
        reviewed: Any,
        *,
        review_status: str,
        review_required: bool,
        corrected_queue_found: bool,
        n_effect_label_changes: int,
    ) -> dict[str, Any]:
        """Build a resilient baseline-vs-reviewed training comparison summary."""

        compare_helper = getattr(self.training_service, "compare_baseline_and_reviewed", None)
        if callable(compare_helper):
            summary = compare_helper(annotated, reviewed)
        else:
            summary = {
                "comparison_scope": "auto_labeled_baseline_vs_reviewed_retrain",
                "baseline_rows": len(self._to_records(annotated)),
                "reviewed_rows": len(self._to_records(reviewed)),
                "baseline_effective_rows": len(self._to_records(annotated)),
                "reviewed_effective_rows": len(self._to_records(reviewed)),
                "datasets_identical": self._to_records(annotated) == self._to_records(reviewed),
                "baseline_status": "not_available_missing_compare_helper",
                "reviewed_status": "not_available_missing_compare_helper",
                "baseline_metrics": {},
                "reviewed_metrics": {},
                "delta_accuracy": None,
                "delta_f1": None,
                "notes": [
                    "Training comparison helper is unavailable for the injected training service, so only the final training metrics are reported.",
                ],
            }

        notes = list(summary.get("notes", [])) if isinstance(summary.get("notes"), list) else []
        if review_required and not corrected_queue_found:
            notes.append("Corrected queue was not loaded in this run, so reviewed retrain still reflects the current auto-labeled dataset.")
        elif review_status == "merged_no_changes":
            notes.append("Corrected queue was merged, but effect labels did not change, so retrain deltas may stay close to zero.")
        elif review_status == "merged" and n_effect_label_changes > 0:
            notes.append("Reviewed retrain includes manual effect-label changes from HITL, so deltas show the impact of post-review supervision.")

        summary["notes"] = notes
        summary["review_status"] = review_status
        summary["review_required"] = review_required
        summary["corrected_queue_found"] = corrected_queue_found
        summary["n_effect_label_changes"] = n_effect_label_changes
        return summary

    def _build_active_learning_comparison_payload(
        self,
        reviewed: Any,
        *,
        entropy_history: list[dict[str, Any]],
        seed_size: int,
        n_iterations: int,
        batch_size: int,
    ) -> dict[str, Any]:
        """Build a resilient entropy-vs-random comparison payload for active learning."""

        compare_helper = getattr(self.active_learning_agent, "compare_strategies", None)
        summarize_helper = getattr(self.active_learning_agent, "summarize_strategy_comparison", None)

        if callable(compare_helper):
            rows = compare_helper(
                reviewed,
                strategies=("entropy", "random"),
                seed_size=seed_size,
                n_iterations=n_iterations,
                batch_size=batch_size,
            )
            if callable(summarize_helper):
                payload = summarize_helper(rows)
            else:
                payload = {
                    "comparison_scope": "entropy_vs_random_active_learning",
                    "strategies": sorted({self._normalize_text(row.get("strategy")) for row in rows if self._normalize_text(row.get("strategy"))}),
                    "rows": rows,
                    "final_by_strategy": {},
                    "delta_accuracy_entropy_minus_random": None,
                    "delta_f1_entropy_minus_random": None,
                    "best_strategy": "",
                    "notes": ["Active-learning comparison rows were produced, but no summary helper is available for this agent."],
                }
            return payload

        return {
            "comparison_scope": "entropy_vs_random_active_learning",
            "strategies": ["entropy"],
            "rows": [
                {
                    "strategy": "entropy",
                    "iteration": row.get("iteration"),
                    "n_labeled": row.get("n_labeled"),
                    "accuracy": row.get("accuracy"),
                    "f1": row.get("f1"),
                }
                for row in entropy_history
            ],
            "final_by_strategy": {"entropy": entropy_history[-1]} if entropy_history else {},
            "delta_accuracy_entropy_minus_random": None,
            "delta_f1_entropy_minus_random": None,
            "best_strategy": "entropy" if entropy_history else "",
            "notes": [
                "The injected active-learning agent does not expose strategy comparison helpers, so only the main entropy run is reported.",
            ],
        }
