"""Unit tests for the richer EDA reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
from src.core.context import PipelineContext
from src.domain import SourceCandidate
from src.services.reporting_service import ReportingService


class _Frame:
    """Tiny dataframe-like helper used to keep the test independent of pandas."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    @property
    def empty(self) -> bool:
        return not self._rows

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        if orient != "records":
            raise ValueError("Only records orientation is supported")
        return [dict(row) for row in self._rows]


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for reporting tests."""

    config = AppConfig(
        project=ProjectConfig(name="reporting-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(effect_labels=["energy", "side_effects", "other"]),
        request=RequestConfig(topic="fitness supplements", domain="supplements", modality="text"),
    )
    return PipelineContext.from_config(config)


def test_eda_context_includes_extended_metrics(tmp_path: Path) -> None:
    """The machine-readable EDA context should expose richer summary blocks."""

    service = ReportingService(_make_context(tmp_path))
    raw = _Frame(
        [
            {"id": "1", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "2", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "3", "source": "Web", "text": "side effect warning", "rating": 1},
        ]
    )
    cleaned = _Frame(
        [
            {"id": "1", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "3", "source": "Web", "text": "side effect warning", "rating": 1},
        ]
    )

    context_path = service.write_eda_context(
        cleaned,
        raw_df_like=raw,
        quality_report={"warnings": ["duplicates removed"]},
    )
    payload = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert payload["n_rows"] == 2
    assert payload["column_count"] == 4
    assert payload["raw_vs_cleaned"]["available"] is True
    assert payload["raw_vs_cleaned"]["raw_rows"] == 3
    assert payload["raw_vs_cleaned"]["cleaned_rows"] == 2
    assert payload["duplicate_summary"]["available"] is True
    assert payload["duplicate_summary"]["duplicate_rows"] == 0
    assert payload["rating_distribution"]["available"] is True
    assert payload["text_length_buckets"]["available"] is True
    assert payload["cleaned_word_cloud"]["available"] is True
    assert payload["cleaned_word_cloud"]["valid_text_rows"] == 2
    assert payload["cleaned_word_cloud"]["terms"][0]["term"] == "energy"
    assert payload["quality_warnings"] == ["duplicates removed"]


def test_eda_context_word_cloud_filters_stop_words_and_digits(tmp_path: Path) -> None:
    """The cleaned word cloud should keep topical tokens and drop stop words or numeric noise."""

    service = ReportingService(_make_context(tmp_path))
    cleaned = _Frame(
        [
            {"id": "1", "source": "HF", "text": "The energy boost was 100 percent real", "rating": 5},
            {"id": "2", "source": "Web", "text": "And the focus boost stayed clean", "rating": 4},
        ]
    )

    context_path = service.write_eda_context(cleaned, raw_df_like=cleaned, quality_report={"warnings": []})
    payload = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))
    terms = [item["term"] for item in payload["cleaned_word_cloud"]["terms"]]

    assert "energy" in terms
    assert "boost" in terms
    assert "the" not in terms
    assert "and" not in terms
    assert "100" not in terms


def test_eda_html_report_is_created(tmp_path: Path) -> None:
    """The HTML EDA export should create a self-contained report file."""

    service = ReportingService(_make_context(tmp_path))
    cleaned = _Frame(
        [
            {"id": "1", "source": "HF", "text": "energy boost", "rating": 5},
            {"id": "2", "source": "Web", "text": "side effect warning", "rating": 1},
        ]
    )

    html_path = service.write_eda_html_report(cleaned, raw_df_like=cleaned, quality_report={"warnings": []})
    html = (tmp_path / html_path).read_text(encoding="utf-8")

    assert "<html" in html.lower()
    assert "EDA Report" in html
    assert "Raw vs cleaned" in html
    assert "Charts" in html


def test_online_governance_report_and_context_are_created(tmp_path: Path) -> None:
    """The governance layer should produce both human-facing and machine-readable artifacts."""

    service = ReportingService(_make_context(tmp_path))
    summary = {
        "remote_sources_enabled": True,
        "active_provider_count": 1,
        "github_auth_mode": "unauthenticated",
        "fallback_strategy": "empty remote shortlist keeps the run stable",
        "notes": ["Configure GITHUB_TOKEN to reduce GitHub Search API rate-limit risk."],
        "providers": [
            {
                "provider_id": "github_repository_search",
                "label": "GitHub repository search API",
                "enabled_in_config": True,
                "active_in_runtime": True,
                "observed_status": "active_no_candidates",
                "discovered_candidates": 0,
                "auth_mode": "unauthenticated",
                "implementation_status": "real_lookup_mvp",
                "rate_limit_guidance": "GitHub Search API is more fragile without GITHUB_TOKEN.",
                "fallback_behavior": "Pipeline continues without remote candidates.",
                "operator_action": "Set GITHUB_TOKEN to reduce GitHub Search API rate-limit risk.",
            }
        ],
    }

    report_path = service.write_online_governance_report(summary)
    context_path = service.write_online_governance_context(summary)

    report = (tmp_path / report_path).read_text(encoding="utf-8")
    context = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert "Online governance and fallback" in report
    assert "github_auth_mode: unauthenticated" in report
    assert "provider_id: github_repository_search" in report
    assert "operator_action: Set GITHUB_TOKEN" in report
    assert context["github_auth_mode"] == "unauthenticated"


def test_review_agreement_report_and_context_are_created(tmp_path: Path) -> None:
    """Agreement reporting should produce both markdown and machine-readable artifacts."""

    service = ReportingService(_make_context(tmp_path))
    summary = {
        "comparison_scope": "auto_vs_human_reviewed_subset",
        "corrected_queue_found": True,
        "n_corrected_rows": 3,
        "n_reviewed_rows": 3,
        "compared_rows": 3,
        "matched_rows": 2,
        "disagreement_rows": 1,
        "agreement": 2 / 3,
        "kappa": 0.4,
        "kappa_status": "computed",
        "auto_label_distribution": {"energy": 1, "side_effects": 1, "other": 1},
        "human_label_distribution": {"energy": 2, "other": 1},
        "disagreement_examples": [
            {
                "id": "2",
                "source": "Web",
                "auto_effect_label": "side_effects",
                "reviewed_effect_label": "energy",
                "confidence": 0.3,
                "text_preview": "rash",
            }
        ],
        "notes": ["This metric measures auto-vs-human agreement on the reviewed subset."],
    }

    report_path = service.write_review_agreement_report(summary)
    context_path = service.write_review_agreement_context(summary)

    report = (tmp_path / report_path).read_text(encoding="utf-8")
    context = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert "Review agreement report" in report
    assert "comparison_scope: auto_vs_human_reviewed_subset" in report
    assert "agreement: 0.667" in report
    assert "kappa: 0.4" in report
    assert "Disagreement examples" in report
    assert context["compared_rows"] == 3
    assert context["kappa_status"] == "computed"


def test_training_comparison_report_and_context_are_created(tmp_path: Path) -> None:
    """Training comparison should produce both markdown and machine-readable artifacts."""

    service = ReportingService(_make_context(tmp_path))
    summary = {
        "comparison_scope": "auto_labeled_baseline_vs_reviewed_retrain",
        "baseline_status": "computed",
        "reviewed_status": "computed",
        "review_status": "merged",
        "corrected_queue_found": True,
        "n_effect_label_changes": 2,
        "datasets_identical": False,
        "delta_accuracy": 0.1,
        "delta_f1": 0.08,
        "baseline_metrics": {"accuracy": 0.7, "f1": 0.68, "n_examples": 12},
        "reviewed_metrics": {"accuracy": 0.8, "f1": 0.76, "n_examples": 12},
        "notes": ["Reviewed retrain includes manual effect-label changes from HITL."],
    }

    report_path = service.write_training_comparison_report(summary)
    context_path = service.write_training_comparison_context(summary)

    report = (tmp_path / report_path).read_text(encoding="utf-8")
    context = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert "Training comparison report" in report
    assert "comparison_scope: auto_labeled_baseline_vs_reviewed_retrain" in report
    assert "delta_accuracy: 0.1" in report
    assert "delta_f1: 0.08" in report
    assert "## Reviewed retrain metrics" in report
    assert context["review_status"] == "merged"


def test_al_comparison_report_and_context_are_created(tmp_path: Path) -> None:
    """AL comparison reporting should produce both markdown and machine-readable artifacts."""

    service = ReportingService(_make_context(tmp_path))
    summary = {
        "comparison_scope": "entropy_vs_random_active_learning",
        "strategies": ["entropy", "random"],
        "best_strategy": "entropy",
        "delta_accuracy_entropy_minus_random": 0.05,
        "delta_f1_entropy_minus_random": 0.04,
        "final_by_strategy": {
            "entropy": {"iteration": 3, "n_labeled": 90, "accuracy": 0.81, "f1": 0.79},
            "random": {"iteration": 3, "n_labeled": 90, "accuracy": 0.76, "f1": 0.75},
        },
        "rows": [
            {"strategy": "entropy", "iteration": 1, "n_labeled": 60, "accuracy": 0.7, "f1": 0.68},
            {"strategy": "random", "iteration": 1, "n_labeled": 60, "accuracy": 0.66, "f1": 0.64},
        ],
        "notes": ["Entropy and random are compared on the same offline text baseline."],
    }

    report_path = service.write_al_comparison_report(summary)
    context_path = service.write_al_comparison_context(summary)

    report = (tmp_path / report_path).read_text(encoding="utf-8")
    context = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert "Active Learning Comparison Report" in report
    assert "best_strategy: entropy" in report
    assert "delta_f1_entropy_minus_random: 0.04" in report
    assert "## Final strategy snapshot" in report
    assert context["best_strategy"] == "entropy"


def test_source_report_and_approval_candidates_include_compliance_fields(tmp_path: Path) -> None:
    """Approval-facing source artifacts should surface license and robots metadata explicitly."""

    service = ReportingService(_make_context(tmp_path))
    source = SourceCandidate(
        "demo_fitness_scrape",
        "scrape",
        "Fitness Supplements Offline Demo",
        "demo://fitness-supplements",
        score=1.0,
        metadata={"html": "<html></html>", "demo_mode": True, "topic": "fitness supplements"},
    )

    report_path = service.write_source_report([source])
    report = (tmp_path / report_path).read_text(encoding="utf-8")
    approval_candidates = json.loads((tmp_path / "data" / "raw" / "approval_candidates.json").read_text(encoding="utf-8"))

    assert approval_candidates == [
        {
            "source_id": "demo_fitness_scrape",
            "source_type": "scrape",
            "title": "Fitness Supplements Offline Demo",
            "uri": "demo://fitness-supplements",
            "score": 1.0,
            "license": "offline_demo_fixture",
            "license_status": "demo_fixture",
            "robots_txt_status": "not_applicable_local_demo",
            "robots_txt_url": "",
            "approval_notes": "offline demo fixture, no external site access",
            "metadata": {"html": "<html></html>", "demo_mode": True, "topic": "fitness supplements"},
        }
    ]
    assert "license: offline_demo_fixture" in report
    assert "robots_txt_status: not_applicable_local_demo" in report
    assert "approval_notes: offline demo fixture, no external site access" in report
    assert "html=" not in report


def test_review_queue_report_and_context_make_hitl_steps_explicit(tmp_path: Path) -> None:
    """The review queue artifacts should tell the reviewer what to do next."""

    service = ReportingService(_make_context(tmp_path))
    review_queue = _Frame(
        [
            {
                "id": "1",
                "source": "HF",
                "text": "energy boost",
                "label": None,
                "effect_label": "energy",
                "confidence": 0.42,
            }
        ]
    )

    report_path = service.write_review_queue_report(review_queue, 0.6, ["energy", "side_effects", "other"])
    context_path = service.write_review_queue_context(review_queue, 0.6, ["energy", "side_effects", "other"])

    report = (tmp_path / report_path).read_text(encoding="utf-8")
    payload = json.loads((tmp_path / context_path).read_text(encoding="utf-8"))

    assert "## Reviewer guide" in report
    assert "## To-do reviewer" in report
    assert "review_queue_corrected.csv" in report
    assert "## Next step" in report
    assert payload["review_required"] is True
    assert payload["current_stage"] == "human_review"
    assert payload["next_step"] == "fill_corrected_queue_and_rerun"
    assert "reviewed_effect_label" in payload["review_columns"]


def test_review_workspace_is_created_and_surfaces_hitl_inputs(tmp_path: Path) -> None:
    """The reviewer workspace should expose queue rows, files, and next-step guidance in one HTML page."""

    service = ReportingService(_make_context(tmp_path))
    review_queue = _Frame(
        [
            {
                "id": "1",
                "source": "HF",
                "text": "energy boost after morning use",
                "label": None,
                "effect_label": "energy",
                "confidence": 0.42,
                "reviewed_effect_label": "",
                "review_comment": "",
                "human_verified": False,
            }
        ]
    )
    service.registry.save_markdown("final_report.md", "# Final Report\n")
    service.registry.save_markdown("reports/review_queue_report.md", "# Review Queue\n")
    service.registry.save_markdown("reports/review_merge_report.md", "# Review Merge\n")
    service.registry.save_json("data/interim/review_queue_context.json", {"current_stage": "human_review"})
    service.registry.save_json("data/interim/review_merge_context.json", {"review_status": "skipped_missing_corrected_queue"})
    service.registry.save_text("data/interim/review_queue.csv", "id\n1\n")

    workspace_path = service.write_review_workspace(
        review_queue,
        0.6,
        ["energy", "side_effects", "other"],
        review_required=True,
        corrected_queue_found=False,
        corrected_queue_path="data/interim/review_queue_corrected.csv",
        review_status="skipped_missing_corrected_queue",
        next_step="human review rerun recommended before final retrain",
        dashboard_path="reports/run_dashboard.html",
        final_report_path="final_report.md",
        review_queue_report_path="reports/review_queue_report.md",
        review_queue_context_path="data/interim/review_queue_context.json",
        review_merge_report_path="reports/review_merge_report.md",
        review_merge_context_path="data/interim/review_merge_context.json",
    )
    html = (tmp_path / workspace_path).read_text(encoding="utf-8")

    assert "HITL Review Workspace" in html
    assert "Waiting for reviewer action" in html
    assert "review_queue_corrected.csv" in html
    assert "reviewed_effect_label" in html
    assert "energy boost after morning use" in html
    assert 'href="../final_report.md"' in html
    assert 'href="review_queue_report.md"' in html


def test_review_merge_report_explains_next_step(tmp_path: Path) -> None:
    """The merge report should explain what happens after HITL merge."""

    service = ReportingService(_make_context(tmp_path))

    report_path = service.write_review_merge_report(
        corrected_queue_found=True,
        corrected_queue_path="data/interim/review_queue_corrected.csv",
        n_corrected_rows=1,
        n_rows_with_reviewed_effect_label=1,
        n_effect_label_changes=1,
        reviewed_effect_labels=["side_effects"],
        review_status="merged",
    )
    report = (tmp_path / report_path).read_text(encoding="utf-8")

    assert "## Next step" in report
    assert "retrain / active learning" in report


def test_run_dashboard_collects_operator_links_and_relative_paths(tmp_path: Path) -> None:
    """The operator dashboard should expose key artifacts from one HTML entry point."""

    service = ReportingService(_make_context(tmp_path))
    service.registry.save_markdown("final_report.md", "# Final Report\n")
    service.registry.save_markdown("reports/source_report.md", "# Source Report\n")
    service.registry.save_markdown("reports/online_governance_report.md", "# Online Governance\n")
    service.registry.save_markdown("reports/review_agreement_report.md", "# Agreement\n")
    service.registry.save_markdown("reports/training_comparison_report.md", "# Training Comparison\n")
    service.registry.save_markdown("reports/al_comparison_report.md", "# Active Learning Comparison\n")
    service.registry.save_text("reports/review_workspace.html", "<html><body>Review Workspace</body></html>")
    service.registry.save_text("reports/eda_report.html", "<html><body>EDA</body></html>")
    service.registry.save_markdown("reports/review_queue_report.md", "# Review Queue\n")
    service.registry.save_markdown("reports/review_merge_report.md", "# Review Merge\n")
    service.registry.save_json("data/interim/review_queue_context.json", {"current_stage": "human_review"})
    service.registry.save_json("data/interim/review_merge_context.json", {"review_status": "skipped_missing_corrected_queue"})
    service.registry.save_json("data/interim/review_agreement_context.json", {"compared_rows": 0})
    service.registry.save_json("data/raw/approval_candidates.json", [{"source_id": "demo"}])
    service.registry.save_json("data/raw/discovered_sources.json", [{"source_id": "demo"}])
    service.registry.save_json("data/raw/online_governance_summary.json", {"github_auth_mode": "unauthenticated"})
    service.registry.save_json(
        "data/interim/eda_context.json",
        {
            "n_rows": 2,
            "cleaned_word_cloud": {
                "available": True,
                "valid_text_rows": 2,
                "token_count": 5,
                "unique_terms": 4,
                "terms": [
                    {"term": "energy", "count": 2, "font_size": 36, "opacity": 1.0},
                    {"term": "boost", "count": 1, "font_size": 20, "opacity": 0.68},
                ],
            },
        },
    )
    service.registry.save_json("data/interim/annotation_trace.json", {"llm_mode": "classify_effect"})
    service.registry.save_json("data/interim/model_metrics.json", {"accuracy": 1.0, "f1": 1.0})
    service.registry.save_text("data/interim/review_queue.csv", "id\n1\n")

    model_path = tmp_path / "data" / "interim" / "model_artifact.pkl"
    vectorizer_path = tmp_path / "data" / "interim" / "vectorizer_artifact.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    vectorizer_path.write_bytes(b"vectorizer")

    summary = {
        "runtime": {
            "requested_mode": "offline_demo",
            "effective_mode": "offline_demo",
            "demo_sources_enabled": True,
            "configured_remote_source_types": ["hf_dataset"],
            "active_remote_source_types": [],
        },
        "dashboard": {
            "dashboard_path": "reports/run_dashboard.html",
            "final_report_path": "final_report.md",
            "pipeline_status": "attention_required",
            "current_stage": "human_review",
            "primary_action": "fill data/interim/review_queue_corrected.csv and rerun pipeline",
            "next_step": "human review rerun recommended before final retrain",
        },
        "sources": {"n_candidates": 1, "source_report_path": "reports/source_report.md"},
        "online_governance": {
            "governance_report_path": "reports/online_governance_report.md",
            "governance_context_path": "data/raw/online_governance_summary.json",
            "active_provider_count": 0,
            "providers_requiring_attention": ["github_repository_search"],
            "github_auth_mode": "unauthenticated",
            "fallback_strategy": "empty remote shortlist keeps the run stable",
        },
        "quality": {"warnings": ["duplicates removed"]},
        "eda": {
            "eda_report_path": "reports/eda_report.md",
            "eda_html_report_path": "reports/eda_report.html",
            "eda_context_path": "data/interim/eda_context.json",
        },
        "annotation": {
            "annotation_report_path": "reports/annotation_report.md",
            "annotation_trace_report_path": "reports/annotation_trace_report.md",
            "annotation_trace_context_path": "data/interim/annotation_trace.json",
            "confidence_threshold": 0.6,
            "n_low_confidence": 1,
        },
        "review": {
            "status": "skipped_missing_corrected_queue",
            "review_queue_rows": 1,
            "review_required": True,
            "review_workspace_path": "reports/review_workspace.html",
            "review_queue_report_path": "reports/review_queue_report.md",
            "review_queue_context_path": "data/interim/review_queue_context.json",
            "review_merge_report_path": "reports/review_merge_report.md",
            "review_merge_context_path": "data/interim/review_merge_context.json",
            "next_step": "human review rerun recommended before final retrain",
        },
        "agreement": {
            "agreement_report_path": "reports/review_agreement_report.md",
            "agreement_context_path": "data/interim/review_agreement_context.json",
            "comparison_scope": "auto_vs_human_reviewed_subset",
            "compared_rows": 0,
            "agreement": None,
            "kappa": None,
            "kappa_status": "not_available_no_compared_rows",
        },
        "training_comparison": {
            "comparison_report_path": "reports/training_comparison_report.md",
            "comparison_context_path": "data/interim/training_comparison.json",
            "comparison_scope": "auto_labeled_baseline_vs_reviewed_retrain",
            "baseline_status": "computed",
            "reviewed_status": "computed",
            "datasets_identical": False,
            "delta_accuracy": 0.05,
            "delta_f1": 0.04,
            "n_effect_label_changes": 1,
        },
        "approval": {
            "approved_sources_path": "data/raw/approved_sources.json",
            "approval_status": "skipped_missing_file",
        },
        "active_learning": {
            "al_report_path": "reports/al_report.md",
            "al_comparison_report_path": "reports/al_comparison_report.md",
            "al_comparison_context_path": "data/interim/al_comparison.json",
        },
        "training": {"accuracy": 1.0, "f1": 1.0},
        "artifacts": {
            "metrics_path": str(tmp_path / "data" / "interim" / "model_metrics.json"),
            "model_path": str(model_path),
            "vectorizer_path": str(vectorizer_path),
        },
    }

    dashboard_path = service.write_run_dashboard(summary)
    html = (tmp_path / dashboard_path).read_text(encoding="utf-8")

    assert "Pipeline Operator Dashboard" in html
    assert "effective_mode: offline_demo" in html
    assert 'href="../final_report.md"' in html
    assert 'href="eda_report.html"' in html
    assert 'href="review_agreement_report.md"' in html
    assert 'href="training_comparison_report.md"' in html
    assert 'href="al_comparison_report.md"' in html
    assert 'href="review_workspace.html"' in html
    assert 'href="online_governance_report.md"' in html
    assert "../data/interim/model_artifact.pkl" in html
    assert "Cleaned word cloud" in html
    assert "Text rows with tokens: 2. Tokens after cleaning: 5. Unique terms: 4." in html
    assert "energy" in html
    assert "Corrected queue CSV" in html
    assert "expected input" in html
    assert "needs action" in html
    assert "GitHub auth mode" in html
