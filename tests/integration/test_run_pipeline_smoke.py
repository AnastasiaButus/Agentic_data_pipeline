"""Smoke test for the CLI entry point and orchestration layer."""

from __future__ import annotations

from pathlib import Path

from src.core.config import load_config
from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.active_learning_agent import ActiveLearningAgent
from src.services.review_queue_service import ReviewQueueService
from src.services.training_service import TrainingService
from src.domain import SourceCandidate
import src.services.pipeline_controller as pipeline_controller_module


def test_run_pipeline_smoke_creates_final_report_and_metrics(tmp_path: Path, monkeypatch) -> None:
    """The CLI should run end-to-end and persist the final report and model metrics."""

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    assert "fitness supplements" in template_text

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"), encoding="utf-8")

    loaded_config = load_config(config_path)
    assert loaded_config.request.topic == "fitness supplements"
    assert loaded_config.annotation.use_llm is False

    from src.services.source_discovery_service import SourceDiscoveryService

    def fake_discovery_run(self):
        return [
            SourceCandidate("scrape-1", "scrape", "Web", str(tmp_path / "sample.html"), metadata={"html": _sample_html()})
        ]

    def fake_collection_run(self, sources):
        return _collection_rows()

    call_order: list[str] = []
    captured_llm_clients: list[object | None] = []

    class FakeAnnotationAgent:
        """Minimal offline annotation stub that records the llm_client wiring."""

        def __init__(self, ctx, llm_client=None, registry=None):
            captured_llm_clients.append(llm_client)
            self.ctx = ctx
            self.llm_client = llm_client

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

    monkeypatch.setattr(pipeline_controller_module, "AnnotationAgent", FakeAnnotationAgent)

    def fake_export_low_confidence_queue(self, df, threshold=0.7):
        call_order.append("export_low_confidence_queue")
        return list(df) if isinstance(df, list) else df.to_dict(orient="records")

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
        return ([{"iteration": 1, "n_labeled": len(records), "accuracy": 1.0, "f1": 1.0}], list(records))

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

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured_llm_clients == [None]
    assert call_order[:3] == ["export_low_confidence_queue", "load_corrected_queue", "merge_reviewed_labels"]
    assert "run_cycle" in call_order
    assert call_order[-1] == "train"
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()


def _sample_html() -> str:
    """Return a tiny HTML fixture used to satisfy the discovery stub."""

    return (
        "<html><body>"
        '<div class="review" data-text="Great energy" data-rating="5">Nice</div>'
        '<div class="review" data-text="Side effect note" data-rating="1">Bad</div>'
        "</body></html>"
    )


def _collection_rows() -> list[dict[str, object]]:
    """Return a small local collection batch used by the smoke test."""

    return [
        {"id": "1", "source": "Web", "text": "Great energy", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
        {"id": "2", "source": "Web", "text": "Side effect note", "label": None, "rating": 1, "created_at": "now", "split": None, "meta_json": "{}"},
        {"id": "3", "source": "Web", "text": "Average experience", "label": None, "rating": 3, "created_at": "now", "split": None, "meta_json": "{}"},
    ]