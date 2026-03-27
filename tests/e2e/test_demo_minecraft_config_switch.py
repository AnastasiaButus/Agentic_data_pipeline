"""End-to-end config switch test for the minecraft demo theme."""

from __future__ import annotations

from pathlib import Path

from src.agents.data_collection_agent import DataCollectionAgent
from src.services.review_queue_service import ReviewQueueService
from src.services.source_discovery_service import SourceDiscoveryService


def test_demo_minecraft_config_switch_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Load the minecraft config through the existing CLI and verify the pipeline still runs."""

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_minecraft.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    assert "minecraft instructions" in template_text

    runtime_config_path = tmp_path / "demo_minecraft.runtime.yaml"
    runtime_config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    from src.core.config import load_config

    loaded_config = load_config(runtime_config_path)
    assert loaded_config.request.topic == "minecraft instructions"
    assert loaded_config.annotation.use_llm is False

    call_order: list[str] = []

    def fake_discovery_run(self):
        call_order.append("discovery")
        return [
            {
                "source_id": "minecraft-dataset-1",
                "source_type": "hf_dataset",
                "title": "Minecraft instructions demo",
                "uri": "local://minecraft-demo",
                "score": 1.0,
                "metadata": {"topic": "minecraft instructions"},
            }
        ]

    def fake_collection_run(self, sources):
        call_order.append("collection")
        assert sources[0]["metadata"]["topic"] == "minecraft instructions"
        return [
            {
                "id": "1",
                "source": "local-minecraft",
                "text": "minecraft energy guide for new players",
                "label": None,
                "rating": None,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
            },
            {
                "id": "2",
                "source": "local-minecraft",
                "text": "minecraft side effect warning after potion use",
                "label": None,
                "rating": None,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
            },
            {
                "id": "3",
                "source": "local-minecraft",
                "text": "minecraft crafting tip with enchanted tools",
                "label": None,
                "rating": None,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
            },
            {
                "id": "4",
                "source": "local-minecraft",
                "text": "minecraft other note without obvious effect",
                "label": None,
                "rating": None,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
            },
        ]

    def fake_load_corrected_queue(self, path=None):
        call_order.append("review_load_corrected")
        return [
            {
                "id": "1",
                "source": "local-minecraft",
                "text": "minecraft energy guide for new players",
                "label": None,
                "rating": None,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "positive",
                "effect_label": "other",
                "confidence": 0.4,
                "reviewed_effect_label": "energy",
                "review_comment": "Reviewed energy guidance",
                "human_verified": True,
            },
            {
                "id": "3",
                "source": "local-minecraft",
                "text": "minecraft crafting tip with enchanted tools",
                "label": None,
                "rating": None,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "neutral",
                "effect_label": "other",
                "confidence": 0.4,
                "reviewed_effect_label": "side_effects",
                "review_comment": "Reviewed side effects",
                "human_verified": True,
            },
        ]

    monkeypatch.setattr(SourceDiscoveryService, "run", fake_discovery_run)
    monkeypatch.setattr(DataCollectionAgent, "run", fake_collection_run)
    monkeypatch.setattr(ReviewQueueService, "load_corrected_queue", fake_load_corrected_queue)

    from run_pipeline import main

    exit_code = main(["--config", str(runtime_config_path)])

    assert exit_code == 0
    assert call_order[:2] == ["discovery", "collection"]
    assert "review_load_corrected" in call_order
    assert "merged" in (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
