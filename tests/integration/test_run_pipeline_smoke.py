"""Smoke test for the CLI entry point and orchestration layer."""

from __future__ import annotations

from pathlib import Path

from src.core.config import load_config


def test_run_pipeline_smoke_creates_final_report_and_metrics(tmp_path: Path) -> None:
    """The CLI should run end-to-end on the persistent fitness demo config without monkeypatch."""

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    assert "fitness supplements" in template_text

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    loaded_config = load_config(config_path)
    assert loaded_config.request.topic == "fitness supplements"
    assert loaded_config.annotation.use_llm is True

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
    assert "Fitness Supplements Offline Demo" in (tmp_path / "data" / "raw" / "discovered_sources.json").read_text(encoding="utf-8")


def test_run_pipeline_smoke_uses_only_approved_sources_when_approval_file_exists(monkeypatch, tmp_path: Path) -> None:
    """An approval file should narrow the collection inputs to the approved shortlist only."""

    import pandas as pd

    from src.agents import data_collection_agent as data_collection_module

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw" / "approved_sources.json").write_text('["missing-source-id"]', encoding="utf-8")

    captured_sources: dict[str, list[object]] = {}

    def fake_collect(self, sources):
        captured_sources["sources"] = list(sources)
        return pd.DataFrame(
            [
                {
                    "id": "1",
                    "source": "approved-demo",
                    "text": "Great product",
                    "label": None,
                    "rating": 5,
                    "created_at": "now",
                    "split": None,
                    "meta_json": "{}",
                    "sentiment_label": None,
                    "effect_label": "energy",
                    "confidence": 1.0,
                },
                {
                    "id": "2",
                    "source": "approved-demo",
                    "text": "Too sweet",
                    "label": None,
                    "rating": 1,
                    "created_at": "now",
                    "split": None,
                    "meta_json": "{}",
                    "sentiment_label": None,
                    "effect_label": "side_effects",
                    "confidence": 1.0,
                },
            ]
        )

    monkeypatch.setattr(data_collection_module.DataCollectionAgent, "run", fake_collect)

    from src.services import training_service as training_module

    monkeypatch.setattr(
        training_module.TrainingService,
        "train",
        lambda self, df: (
            {
                "model_path": str(tmp_path / "model.pkl"),
                "vectorizer_path": str(tmp_path / "vectorizer.pkl"),
                "metrics_path": str(tmp_path / "metrics.json"),
            },
            {"accuracy": 1.0, "f1": 1.0},
        ),
    )

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured_sources["sources"] == []
    assert (tmp_path / "final_report.md").exists()