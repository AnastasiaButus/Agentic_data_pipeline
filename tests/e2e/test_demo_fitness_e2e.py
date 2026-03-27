"""End-to-end coverage for the demo fitness pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from src.core.config import load_config


def test_demo_fitness_e2e_pipeline_runs_and_produces_reports(tmp_path: Path) -> None:
    """The full demo pipeline should run locally on the persistent fitness config without monkeypatch."""

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
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
    assert "Fitness Supplements Offline Demo" in (tmp_path / "data" / "raw" / "discovered_sources.json").read_text(encoding="utf-8")
    assert captured_report["summary"]["approval"]["approval_status"] == "skipped_missing_file"