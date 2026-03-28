"""Integration coverage for the offline active learning pipeline."""

from __future__ import annotations

from pathlib import Path

from src.agents.active_learning_agent import ActiveLearningAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.services.reporting_service import ReportingService


class FakeRegistry:
    """Capture artifact writes without touching the filesystem."""

    def __init__(self) -> None:
        self.markdown_writes: dict[str, str] = {}
        self.json_writes: dict[str, object] = {}

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        return Path(path)

    def save_markdown(self, path: str | Path, content: str) -> Path:
        self.markdown_writes[str(path)] = content
        return Path(path)

    def save_json(self, path: str | Path, payload: object) -> Path:
        self.json_writes[str(path)] = payload
        return Path(path)


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for the integration test."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def _synthetic_records() -> list[dict[str, object]]:
    """Create a larger synthetic dataset so the AL loop can run multiple iterations."""

    rows: list[dict[str, object]] = []
    labels = ["energy", "side_effects", "other"]
    for index in range(120):
        label = labels[index % len(labels)]
        rows.append(
            {
                "id": str(index + 1),
                "source": "HF",
                "text": f"Synthetic review {index} talking about {label} and supplements",
                "label": None,
                "rating": 5 if label == "energy" else (1 if label == "side_effects" else 3),
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": None,
                "effect_label": label,
                "confidence": 1.0,
            }
        )
    return rows


def test_al_pipeline_runs_multiple_iterations_with_entropy_and_random(tmp_path: Path) -> None:
    """The offline AL pipeline should run multiple iterations and support entropy/random comparison."""

    registry = FakeRegistry()
    context = _make_context(tmp_path)
    entropy_agent = ActiveLearningAgent(context, registry=registry, random_state=7)
    random_agent = ActiveLearningAgent(context, registry=registry, random_state=7)
    reporting = ReportingService(context, registry=registry)
    records = _synthetic_records()

    entropy_history, entropy_labeled = entropy_agent.run_cycle(records, strategy="entropy", seed_size=50, n_iterations=5, batch_size=20)
    random_history, random_labeled = random_agent.run_cycle(records, strategy="random", seed_size=50, n_iterations=5, batch_size=20)
    comparison_rows = entropy_agent.compare_strategies(records, strategies=("entropy", "random"), seed_size=50, n_iterations=2, batch_size=20)
    comparison_summary = entropy_agent.summarize_strategy_comparison(comparison_rows)
    comparison_report = reporting.write_al_comparison_report(comparison_summary)
    comparison_context = reporting.write_al_comparison_context(comparison_summary)

    assert len(entropy_history) >= 2
    assert len(random_history) >= 2
    assert all({"iteration", "n_labeled", "accuracy", "f1"}.issubset(row.keys()) for row in entropy_history)
    assert all({"iteration", "n_labeled", "accuracy", "f1"}.issubset(row.keys()) for row in random_history)
    assert len(entropy_labeled) >= 50
    assert len(random_labeled) >= 50
    assert comparison_rows
    assert {"entropy", "random"}.issubset({row["strategy"] for row in comparison_rows})
    assert comparison_report == "reports/al_comparison_report.md"
    assert Path(comparison_context).as_posix() == "data/interim/al_comparison.json"
    markdown = registry.markdown_writes[comparison_report]
    assert "best_strategy" in markdown
    assert "strategy" in markdown
    assert "iteration" in markdown
    assert "n_labeled" in markdown
    assert "accuracy" in markdown
    assert "f1" in markdown
