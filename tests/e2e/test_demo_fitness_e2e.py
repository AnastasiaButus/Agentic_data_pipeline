"""End-to-end coverage for the demo fitness pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import load_config


def test_demo_fitness_e2e_pipeline_runs_and_produces_reports(monkeypatch, tmp_path: Path) -> None:
    """The full demo pipeline should run locally on the persistent fitness config without monkeypatch."""

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    runtime_config_path = tmp_path / "demo_fitness.runtime.yaml"
    runtime_config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    loaded_config = load_config(runtime_config_path)
    assert loaded_config.request.topic == "fitness supplements"
    assert loaded_config.annotation.use_llm is True
    assert loaded_config.annotation.llm_provider == "mock"
    assert loaded_config.runtime.mode == "offline_demo"

    from src.services import reporting_service as reporting_module

    captured_report: dict[str, object] = {}
    original_write_final_report = reporting_module.ReportingService.write_final_report

    def capture_final_report(self, summary):
        captured_report["summary"] = summary
        return original_write_final_report(self, summary)

    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(reporting_module.ReportingService, "write_final_report", capture_final_report)

    from run_pipeline import main

    try:
        exit_code = main(["--config", str(runtime_config_path)])
    finally:
        monkeypatch.undo()

    assert exit_code == 0
    assert (tmp_path / "final_report.md").exists()
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "## Approval" in final_report
    assert "## Dashboard" in final_report
    assert "## EDA" in final_report
    assert "## Annotation" in final_report
    assert "## Agreement" in final_report
    assert "## Training Comparison" in final_report
    assert "eda_report_path" in final_report
    assert "eda_context_path" in final_report
    assert "annotation_trace_report_path" in final_report
    assert "annotation_trace_context_path" in final_report
    assert "approval_status: skipped_missing_file" in final_report
    assert "source_approval_workspace_path: reports/source_approval_workspace.html" in final_report
    assert "governance_report_path: reports/online_governance_report.md" in final_report
    assert "review_workspace_path: reports/review_workspace.html" in final_report
    assert "agreement_report_path: reports/review_agreement_report.md" in final_report
    assert "comparison_report_path: reports/training_comparison_report.md" in final_report
    assert "al_comparison_report_path: reports/al_comparison_report.md" in final_report
    assert "review_merge_report_path" in final_report
    approval_candidates = json.loads((tmp_path / "data" / "raw" / "approval_candidates.json").read_text(encoding="utf-8"))
    assert isinstance(approval_candidates, list)
    assert len(approval_candidates) == 1
    assert approval_candidates[0]["source_id"] == "demo_fitness_scrape"
    assert approval_candidates[0]["title"] == "Fitness Supplements Offline Demo"
    assert approval_candidates[0]["license"] == "offline_demo_fixture"
    assert approval_candidates[0]["license_status"] == "demo_fixture"
    assert approval_candidates[0]["robots_txt_status"] == "not_applicable_local_demo"
    review_queue_report = (tmp_path / "reports" / "review_queue_report.md").read_text(encoding="utf-8")
    assert "# Очередь ручной проверки" in review_queue_report
    assert "ручной проверки после авторазметки" in review_queue_report
    review_workspace = (tmp_path / "reports" / "review_workspace.html").read_text(encoding="utf-8")
    assert "HITL Review Workspace" in review_workspace
    assert "review_queue_corrected.csv" in review_workspace
    assert "Interactive review editor" in review_workspace
    assert "Download corrected queue CSV" in review_workspace
    assert "copy-corrected-path" in review_workspace
    agreement_report = (tmp_path / "reports" / "review_agreement_report.md").read_text(encoding="utf-8")
    assert "Review agreement report" in agreement_report
    assert "compared_rows: 0" in agreement_report
    training_comparison_context = json.loads((tmp_path / "data" / "interim" / "training_comparison.json").read_text(encoding="utf-8"))
    assert training_comparison_context["comparison_scope"] == "auto_labeled_baseline_vs_reviewed_retrain"
    assert training_comparison_context["datasets_identical"] is True
    al_comparison_context = json.loads((tmp_path / "data" / "interim" / "al_comparison.json").read_text(encoding="utf-8"))
    assert {"entropy", "random"}.issubset(set(al_comparison_context["strategies"]))
    review_queue_context = json.loads((tmp_path / "data" / "interim" / "review_queue_context.json").read_text(encoding="utf-8"))
    assert review_queue_context["confidence_threshold"] == loaded_config.annotation.confidence_threshold
    assert review_queue_context["n_rows"] >= 0
    assert review_queue_context["label_options"] == loaded_config.annotation.effect_labels
    annotation_trace_report = (tmp_path / "reports" / "annotation_trace_report.md").read_text(encoding="utf-8")
    assert "Трассировка annotation contract" in annotation_trace_report
    assert "Верни только JSON" in annotation_trace_report
    annotation_trace_context = json.loads((tmp_path / "data" / "interim" / "annotation_trace.json").read_text(encoding="utf-8"))
    assert "prompt_contract" in annotation_trace_context
    assert "parser_contract" in annotation_trace_context
    assert annotation_trace_context["llm_mode"] == "classify_effect"
    assert annotation_trace_context["n_rows"] >= 0
    eda_report = (tmp_path / "reports" / "eda_report.md").read_text(encoding="utf-8")
    assert "EDA-пакет" in eda_report
    assert "Это расширенный честный EDA-отчет" in eda_report
    eda_context = json.loads((tmp_path / "data" / "interim" / "eda_context.json").read_text(encoding="utf-8"))
    assert "n_rows" in eda_context
    assert "columns" in eda_context
    assert "cleaned_word_cloud" in eda_context
    assert "missing_values_summary" in eda_context
    merge_report = (tmp_path / "reports" / "review_merge_report.md").read_text(encoding="utf-8")
    assert "Merge не выполнен" in merge_report
    assert "corrected queue отсутствует" in merge_report
    merge_context = json.loads((tmp_path / "data" / "interim" / "review_merge_context.json").read_text(encoding="utf-8"))
    assert merge_context["corrected_queue_found"] is False
    assert merge_context["n_corrected_rows"] == 0
    assert merge_context["n_effect_label_changes"] == 0
    source_report = (tmp_path / "reports" / "source_report.md").read_text(encoding="utf-8")
    source_approval_workspace = (tmp_path / "reports" / "source_approval_workspace.html").read_text(encoding="utf-8")
    assert "Короткий shortlist источников" in source_report
    assert "ручного просмотра и одобрения" in source_report
    assert "Fitness Supplements Offline Demo" in source_report
    assert "license: offline_demo_fixture" in source_report
    assert "robots_txt_status: not_applicable_local_demo" in source_report
    assert "Source Approval Workspace" in source_approval_workspace
    assert "Download approved_sources.json" in source_approval_workspace
    assert "demo_fitness_scrape" in source_approval_workspace
    governance_report = (tmp_path / "reports" / "online_governance_report.md").read_text(encoding="utf-8")
    assert "Online governance and fallback" in governance_report
    assert "configured_but_inactive_for_runtime" in governance_report
    dashboard_html = (tmp_path / "reports" / "run_dashboard.html").read_text(encoding="utf-8")
    assert "Pipeline Operator Dashboard" in dashboard_html
    assert "../final_report.md" in dashboard_html
    assert "review_agreement_report.md" in dashboard_html
    assert "training_comparison_report.md" in dashboard_html
    assert "al_comparison_report.md" in dashboard_html
    assert "review_workspace.html" in dashboard_html
    assert "source_approval_workspace.html" in dashboard_html
    assert "online_governance_report.md" in dashboard_html
    assert "Cleaned word cloud" in dashboard_html
    assert "HITL control center" in dashboard_html
    assert "LLM annotation center" in dashboard_html
    assert "offline_mock_llm_active" in dashboard_html
    assert "review_queue_corrected.csv" in dashboard_html
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
    assert "Fitness Supplements Offline Demo" in (tmp_path / "data" / "raw" / "discovered_sources.json").read_text(encoding="utf-8")
    assert captured_report["summary"]["runtime"]["effective_mode"] == "offline_demo"
    assert captured_report["summary"]["runtime"]["demo_sources_enabled"] is True
    assert captured_report["summary"]["runtime"]["remote_sources_enabled"] is False
    assert captured_report["summary"]["runtime"]["active_remote_source_types"] == []
    assert captured_report["summary"]["dashboard"]["dashboard_path"] == "reports/run_dashboard.html"
    assert captured_report["summary"]["dashboard"]["final_report_path"] == "final_report.md"
    assert captured_report["summary"]["approval"]["approval_status"] == "skipped_missing_file"
    assert captured_report["summary"]["approval"]["source_approval_workspace_path"] == "reports/source_approval_workspace.html"
    assert captured_report["summary"]["active_learning"]["al_comparison_report_path"] == "reports/al_comparison_report.md"
    assert captured_report["summary"]["training_comparison"]["comparison_report_path"] == "reports/training_comparison_report.md"
    assert Path(captured_report["summary"]["annotation"]["annotation_trace_report_path"]).as_posix() == "reports/annotation_trace_report.md"
    assert Path(captured_report["summary"]["annotation"]["annotation_trace_context_path"]).as_posix() == "data/interim/annotation_trace.json"
    assert Path(captured_report["summary"]["eda"]["eda_report_path"]).as_posix() == "reports/eda_report.md"
    assert Path(captured_report["summary"]["eda"]["eda_context_path"]).as_posix() == "data/interim/eda_context.json"
    
