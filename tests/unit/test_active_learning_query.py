"""Tests for active learning query and split behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.active_learning_agent import ActiveLearningAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext


class FakeRegistry:
    """Capture artifact writes without touching the filesystem."""

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        return Path(path)

    def save_markdown(self, path: str | Path, content: str) -> Path:
        return Path(path)


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for the active learning tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def _synthetic_records(total_rows: int = 90) -> list[dict[str, object]]:
    """Create a small multiclass text dataset for offline active learning tests."""

    rows: list[dict[str, object]] = []
    labels = ["energy", "side_effects", "other"]
    for index in range(total_rows):
        label = labels[index % len(labels)]
        rows.append(
            {
                "id": str(index + 1),
                "source": "HF",
                "text": f"Review {index} about {label} and supplements",
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


def test_split_seed_and_pool_handles_small_dataset(tmp_path: Path) -> None:
    """Small datasets should be handled without crashing and should keep all rows in the seed set."""

    agent = ActiveLearningAgent(_make_context(tmp_path), registry=FakeRegistry(), random_state=13)
    small_records = _synthetic_records(total_rows=4)

    seed_df, pool_df = agent.split_seed_and_pool(small_records, seed_size=50)

    assert len(seed_df) == 4
    assert pool_df.empty


def test_feature_target_mapping_uses_text_and_effect_label(tmp_path: Path) -> None:
    """The AL baseline should be explicitly mapped to text -> effect_label."""

    agent = ActiveLearningAgent(_make_context(tmp_path), registry=FakeRegistry(), random_state=13)

    assert agent.feature_column == "text"
    assert agent.target_column == "effect_label"


def test_query_entropy_and_random_and_unknown_strategy(tmp_path: Path) -> None:
    """Querying should support entropy and random strategies and reject unknown strategies."""

    agent = ActiveLearningAgent(_make_context(tmp_path), registry=FakeRegistry(), random_state=13)
    records = _synthetic_records()
    seed_df, pool_df = agent.split_seed_and_pool(records, seed_size=50)
    model_bundle = agent.fit(seed_df)

    entropy_indices = agent.query(model_bundle, pool_df, strategy="entropy", batch_size=20)
    random_indices = agent.query(model_bundle, pool_df, strategy="random", batch_size=20)

    assert len(entropy_indices) == 20
    assert len(random_indices) == 20
    assert len(set(random_indices)) == 20

    with pytest.raises(ValueError):
        agent.query(model_bundle, pool_df, strategy="unsupported", batch_size=20)


def test_run_cycle_returns_history_with_expected_metrics(tmp_path: Path) -> None:
    """The active learning cycle should return history rows with the expected metric keys."""

    agent = ActiveLearningAgent(_make_context(tmp_path), registry=FakeRegistry(), random_state=13)
    history, labeled_df = agent.run_cycle(_synthetic_records(), strategy="entropy", seed_size=50, n_iterations=2, batch_size=20)

    assert len(history) >= 2
    assert {"iteration", "n_labeled", "accuracy", "f1"}.issubset(history[0].keys())
    assert len(labeled_df) >= 50


def test_compare_strategies_returns_entropy_and_random_rows(tmp_path: Path) -> None:
    """Strategy comparison should flatten per-iteration rows for entropy and random."""

    agent = ActiveLearningAgent(_make_context(tmp_path), registry=FakeRegistry(), random_state=13)

    rows = agent.compare_strategies(_synthetic_records(), strategies=("entropy", "random"), seed_size=50, n_iterations=2, batch_size=20)

    assert rows
    assert {"entropy", "random"}.issubset({row["strategy"] for row in rows})
    for row in rows:
        assert {"strategy", "iteration", "n_labeled", "accuracy", "f1"}.issubset(row.keys())


def test_summarize_strategy_comparison_returns_final_deltas(tmp_path: Path) -> None:
    """The AL comparison summary should expose final metrics and entropy-vs-random deltas."""

    agent = ActiveLearningAgent(_make_context(tmp_path), registry=FakeRegistry(), random_state=13)
    rows = agent.compare_strategies(_synthetic_records(), strategies=("entropy", "random"), seed_size=50, n_iterations=2, batch_size=20)

    summary = agent.summarize_strategy_comparison(rows)

    assert summary["comparison_scope"] == "entropy_vs_random_active_learning"
    assert {"entropy", "random"}.issubset(summary["strategies"])
    assert "entropy" in summary["final_by_strategy"]
    assert "random" in summary["final_by_strategy"]
    assert "best_strategy" in summary
    assert "delta_f1_entropy_minus_random" in summary
