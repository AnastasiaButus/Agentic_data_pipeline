"""Top-level orchestration for the demo fitness supplements pipeline."""

from __future__ import annotations

from typing import Any

from src.agents.active_learning_agent import ActiveLearningAgent
from src.agents.annotation_agent import AnnotationAgent
from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.data_quality_agent import DataQualityAgent
from src.core.context import PipelineContext
from src.providers.llm.mock_llm import MockLLM
from src.services.review_queue_service import ReviewQueueService
from src.services.reporting_service import ReportingService
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
        self.annotation_agent = annotation_agent if annotation_agent is not None else AnnotationAgent(ctx, llm_client=MockLLM())
        self.review_queue_service = review_queue_service if review_queue_service is not None else ReviewQueueService(ctx)
        self.active_learning_agent = active_learning_agent if active_learning_agent is not None else ActiveLearningAgent(ctx)
        self.training_service = training_service if training_service is not None else TrainingService(ctx)
        self.reporting_service = reporting_service if reporting_service is not None else ReportingService(ctx)

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end and return a compact run summary."""

        sources = self.discovery_service.run()
        source_report_path = self.reporting_service.write_source_report(sources)

        collected = self.collection_agent.run(sources)
        quality_report = self.quality_agent.detect_issues(collected)
        cleaned = self.quality_agent.run(collected)
        quality_report_path = self.reporting_service.write_quality_report(quality_report)

        annotated = self.annotation_agent.auto_label(cleaned)
        annotation_summary = self.annotation_agent.check_quality(annotated)
        annotation_report_path = self.reporting_service.write_annotation_report(annotated, annotation_summary)

        # Export the reviewer queue before consuming corrected labels so HITL semantics stay consistent.
        review_queue = self.review_queue_service.export_low_confidence_queue(annotated)

        reviewed = annotated
        review_status = "skipped_missing_corrected_queue"
        try:
            corrected_queue = self.review_queue_service.load_corrected_queue()
        except FileNotFoundError:
            corrected_queue = None

        if corrected_queue is not None:
            reviewed = self.review_queue_service.merge_reviewed_labels(annotated, corrected_queue)
            review_status = "merged"

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

        artifacts, training_metrics = self.training_service.train(al_labeled)

        final_report_path = self.reporting_service.write_final_report(
            {
                "sources": {
                    "n_candidates": len(sources),
                    "source_report_path": source_report_path,
                },
                "quality": {
                    "quality_report_path": quality_report_path,
                    "warnings": getattr(quality_report, "warnings", []),
                },
                "annotation": {
                    "annotation_report_path": annotation_report_path,
                    "confidence_threshold": annotation_summary.get("confidence_threshold"),
                    "n_low_confidence": annotation_summary.get("n_low_confidence"),
                },
                "review": {
                    "status": review_status,
                    "review_queue_rows": len(self._to_records(review_queue)),
                },
                "active_learning": {
                    "al_report_path": al_report_path,
                    "history": al_history,
                },
                "training": training_metrics,
                "artifacts": artifacts,
            }
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
                "quality_report": quality_report_path,
                "annotation_report": annotation_report_path,
                "al_report": al_report_path,
                "final_report": final_report_path,
            },
            "review_status": review_status,
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