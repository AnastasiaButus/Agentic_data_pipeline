"""End-to-end coverage for the demo fitness pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.active_learning_agent import ActiveLearningAgent
from src.domain import SourceCandidate
from src.services.pipeline_controller import PipelineController
from src.services.review_queue_service import ReviewQueueService
from src.services.training_service import TrainingService
from src.services.source_discovery_service import SourceDiscoveryService


def test_demo_fitness_e2e_pipeline_runs_and_produces_reports(tmp_path: Path, monkeypatch) -> None:
    """The full demo pipeline should run locally, perform HITL merge, train, and write reports."""

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig, RequestConfig
    from src.core.context import PipelineContext

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        request=RequestConfig(topic="fitness supplements", modality="text", task_type="classification", domain="supplements"),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(use_llm=False, confidence_threshold=0.6),
    )
    ctx = PipelineContext.from_config(config)

    def fake_discovery_run(self):
        return [SourceCandidate("scrape-1", "scrape", "Web", str(tmp_path / "sample.html"), metadata={"html": _sample_html()})]

    def fake_collection_run(self, sources):
        return _collection_rows()

    class InjectedAnnotationAgent:
        """Offline annotation stub injected into the controller to prove it is preserved."""

        def auto_label(self, df):
            rows = list(df) if isinstance(df, list) else df.to_dict(orient="records")
            output = []
            for row in rows:
                text = str(row.get("text", "")).lower()
                effect_label = "energy" if "energy" in text else ("side_effects" if "side effect" in text else "other")
                sentiment_label = "positive" if effect_label == "energy" else ("negative" if effect_label == "side_effects" else "neutral")
                output.append(
                    {
                        **row,
                        "sentiment_label": sentiment_label,
                        "effect_label": effect_label,
                        "confidence": 0.5,
                    }
                )
            return output

        def check_quality(self, df_labeled):
            rows = list(df_labeled) if isinstance(df_labeled, list) else df_labeled.to_dict(orient="records")
            return {"confidence_threshold": 0.6, "n_low_confidence": len(rows), "n_rows": len(rows)}

    call_order: list[str] = []

    def fake_load_corrected_queue(self, path=None):
        call_order.append("load_corrected_queue")
        return [
            {
                "id": "1",
                "source": "Web",
                "text": "Great energy",
                "label": None,
                "rating": 5,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "positive",
                "effect_label": "energy",
                "confidence": 0.4,
                "reviewed_effect_label": "other",
                "review_comment": "Reviewed",
                "human_verified": True,
            }
        ]

    def fake_export_low_confidence_queue(self, df, threshold=0.7):
        call_order.append("export_low_confidence_queue")
        return _collection_rows()

    def fake_merge_reviewed_labels(self, original_df, corrected_df):
        call_order.append("merge_reviewed_labels")
        return [
            {
                "id": "1",
                "source": "Web",
                "text": "Great energy",
                "label": None,
                "rating": 5,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "positive",
                "effect_label": "other",
                "confidence": 1.0,
                "reviewed_effect_label": "other",
                "review_comment": "Reviewed",
                "human_verified": True,
            },
            {
                "id": "2",
                "source": "Web",
                "text": "Side effect note",
                "label": None,
                "rating": 1,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "negative",
                "effect_label": "side_effects",
                "confidence": 0.9,
                "reviewed_effect_label": "side_effects",
                "review_comment": "Reviewed",
                "human_verified": True,
            },
        ]

    def fake_run_cycle(self, df, strategy="entropy", seed_size=50, n_iterations=5, batch_size=20):
        call_order.append("run_cycle")
        records = list(df) if isinstance(df, list) else df.to_dict(orient="records")
        assert any(row.get("reviewed_effect_label") == "other" for row in records)
        return (
            [
                {"iteration": 1, "n_labeled": len(records), "accuracy": 1.0, "f1": 1.0},
                {"iteration": 2, "n_labeled": len(records), "accuracy": 1.0, "f1": 1.0},
            ],
            list(records),
        )

    def fake_train(self, df):
        call_order.append("train")
        records = list(df) if isinstance(df, list) else df.to_dict(orient="records")
        assert any(row.get("reviewed_effect_label") == "other" for row in records)
        metrics_path = tmp_path / "data" / "interim" / "model_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("{}", encoding="utf-8")
        return (
            {
                "model_path": str(tmp_path / "data" / "interim" / "model_artifact.pkl"),
                "vectorizer_path": str(tmp_path / "data" / "interim" / "vectorizer_artifact.pkl"),
                "metrics_path": str(metrics_path),
            },
            {"accuracy": 1.0, "f1": 1.0},
        )

    monkeypatch.setattr(SourceDiscoveryService, "run", fake_discovery_run)
    monkeypatch.setattr(DataCollectionAgent, "run", fake_collection_run)
    monkeypatch.setattr(ReviewQueueService, "export_low_confidence_queue", fake_export_low_confidence_queue)
    monkeypatch.setattr(ReviewQueueService, "load_corrected_queue", fake_load_corrected_queue)
    monkeypatch.setattr(ReviewQueueService, "merge_reviewed_labels", fake_merge_reviewed_labels)
    monkeypatch.setattr(ActiveLearningAgent, "run_cycle", fake_run_cycle)
    monkeypatch.setattr(TrainingService, "train", fake_train)

    injected_annotation_agent = InjectedAnnotationAgent()
    controller = PipelineController(ctx, annotation_agent=injected_annotation_agent)

    assert controller.annotation_agent is injected_annotation_agent
    summary = controller.run()

    assert summary["review_status"] == "merged"
    assert controller.annotation_agent is injected_annotation_agent
    assert call_order[:3] == ["export_low_confidence_queue", "load_corrected_queue", "merge_reviewed_labels"]
    assert "run_cycle" in call_order
    assert call_order[-1] == "train"
    assert len(summary["active_learning_history"]) >= 2
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()


def _sample_html() -> str:
    """Return a tiny HTML fixture used by the discovery stub."""

    return (
        "<html><body>"
        '<div class="review" data-text="Great energy" data-rating="5">Nice</div>'
        '<div class="review" data-text="Side effect note" data-rating="1">Bad</div>'
        "</body></html>"
    )


def _collection_rows() -> list[dict[str, object]]:
    """Return a small local collection batch used by the e2e test."""

    return [
        {"id": "1", "source": "Web", "text": "Great energy", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
        {"id": "2", "source": "Web", "text": "Side effect note", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        {"id": "3", "source": "Web", "text": "Average experience", "label": None, "rating": 3, "created_at": "now", "split": None, "meta_json": "{}"},
        {"id": "4", "source": "Web", "text": "More energy", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
        {"id": "5", "source": "Web", "text": "Another side effect", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        {"id": "6", "source": "Web", "text": "Nothing special", "label": None, "rating": 3, "created_at": "now", "split": None, "meta_json": "{}"},
    ]