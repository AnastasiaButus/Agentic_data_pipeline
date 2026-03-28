"""Unit tests for CLI ergonomics around artifact discovery and auto-open behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import run_pipeline


def _fake_context(root_dir: Path) -> SimpleNamespace:
    """Build a tiny context-like object exposing only the path contract used by the CLI."""

    return SimpleNamespace(paths=SimpleNamespace(root_dir=root_dir))


def test_main_prints_operator_facing_paths(monkeypatch, tmp_path: Path, capsys) -> None:
    """The CLI should print the main artifact paths so first-time users know what to open."""

    fake_config = object()
    fake_context = _fake_context(tmp_path)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_main_prints_operator_facing_paths")

    class FakePipelineContext:
        @classmethod
        def from_config(cls, config):
            assert config is fake_config
            return fake_context

    class FakeController:
        def __init__(self, ctx):
            assert ctx is fake_context

        def run(self):
            return {
                "reports": {
                    "dashboard": "reports/run_dashboard.html",
                    "final_report": "final_report.md",
                    "eda_html_report": "reports/eda_report.html",
                    "review_workspace": "reports/review_workspace.html",
                },
                "review_status": "skipped_missing_corrected_queue",
                "runtime_mode": "offline_demo",
            }

    monkeypatch.setattr(run_pipeline, "load_config", lambda path: fake_config)
    monkeypatch.setattr(run_pipeline, "PipelineContext", FakePipelineContext)
    monkeypatch.setattr(run_pipeline, "PipelineController", FakeController)

    exit_code = run_pipeline.main(["--config", str(tmp_path / "demo.yaml")])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Pipeline run completed." in captured.out
    assert "Runtime mode: offline_demo" in captured.out
    assert str((tmp_path / "reports" / "run_dashboard.html").resolve()) in captured.out
    assert (tmp_path / "reports" / "run_dashboard.html").resolve().as_uri() in captured.out
    assert str((tmp_path / "final_report.md").resolve()) in captured.out
    assert (tmp_path / "final_report.md").resolve().as_uri() in captured.out
    assert str((tmp_path / "reports" / "review_workspace.html").resolve()) in captured.out
    assert (tmp_path / "reports" / "review_workspace.html").resolve().as_uri() in captured.out
    assert "fill review_queue_corrected.csv and rerun the pipeline" in captured.out


def test_main_opens_requested_artifacts(monkeypatch, tmp_path: Path) -> None:
    """The CLI should optionally open the generated dashboard and review workspace."""

    fake_config = object()
    fake_context = _fake_context(tmp_path)
    opened_urls: list[str] = []

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "run_dashboard.html").write_text("<html>dashboard</html>", encoding="utf-8")
    (reports_dir / "review_workspace.html").write_text("<html>review</html>", encoding="utf-8")

    class FakePipelineContext:
        @classmethod
        def from_config(cls, config):
            assert config is fake_config
            return fake_context

    class FakeController:
        def __init__(self, ctx):
            assert ctx is fake_context

        def run(self):
            return {
                "reports": {
                    "dashboard": "reports/run_dashboard.html",
                    "review_workspace": "reports/review_workspace.html",
                },
                "review_status": "merged",
                "runtime_mode": "offline_demo",
            }

    monkeypatch.setattr(run_pipeline, "load_config", lambda path: fake_config)
    monkeypatch.setattr(run_pipeline, "PipelineContext", FakePipelineContext)
    monkeypatch.setattr(run_pipeline, "PipelineController", FakeController)
    monkeypatch.setattr(run_pipeline.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    exit_code = run_pipeline.main(
        [
            "--config",
            str(tmp_path / "demo.yaml"),
            "--open-dashboard",
            "--open-review-workspace",
        ]
    )

    assert exit_code == 0
    assert (reports_dir / "run_dashboard.html").resolve().as_uri() in opened_urls
    assert (reports_dir / "review_workspace.html").resolve().as_uri() in opened_urls


def test_main_auto_opens_dashboard_by_default_outside_pytest(monkeypatch, tmp_path: Path) -> None:
    """A normal CLI run should auto-open the dashboard so the interface is attached to the run."""

    fake_config = object()
    fake_context = _fake_context(tmp_path)
    opened_urls: list[str] = []

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "run_dashboard.html").write_text("<html>dashboard</html>", encoding="utf-8")

    class FakePipelineContext:
        @classmethod
        def from_config(cls, config):
            assert config is fake_config
            return fake_context

    class FakeController:
        def __init__(self, ctx):
            assert ctx is fake_context

        def run(self):
            return {
                "reports": {
                    "dashboard": "reports/run_dashboard.html",
                },
                "review_status": "merged",
                "runtime_mode": "offline_demo",
            }

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(run_pipeline, "load_config", lambda path: fake_config)
    monkeypatch.setattr(run_pipeline, "PipelineContext", FakePipelineContext)
    monkeypatch.setattr(run_pipeline, "PipelineController", FakeController)
    monkeypatch.setattr(run_pipeline.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    exit_code = run_pipeline.main(["--config", str(tmp_path / "demo.yaml")])

    assert exit_code == 0
    assert opened_urls == [(reports_dir / "run_dashboard.html").resolve().as_uri()]


def test_main_no_open_dashboard_disables_default_auto_open(monkeypatch, tmp_path: Path) -> None:
    """The operator should still be able to suppress auto-open when needed."""

    fake_config = object()
    fake_context = _fake_context(tmp_path)
    opened_urls: list[str] = []

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "run_dashboard.html").write_text("<html>dashboard</html>", encoding="utf-8")

    class FakePipelineContext:
        @classmethod
        def from_config(cls, config):
            assert config is fake_config
            return fake_context

    class FakeController:
        def __init__(self, ctx):
            assert ctx is fake_context

        def run(self):
            return {
                "reports": {
                    "dashboard": "reports/run_dashboard.html",
                },
                "review_status": "merged",
                "runtime_mode": "offline_demo",
            }

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(run_pipeline, "load_config", lambda path: fake_config)
    monkeypatch.setattr(run_pipeline, "PipelineContext", FakePipelineContext)
    monkeypatch.setattr(run_pipeline, "PipelineController", FakeController)
    monkeypatch.setattr(run_pipeline.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    exit_code = run_pipeline.main(["--config", str(tmp_path / "demo.yaml"), "--no-open-dashboard"])

    assert exit_code == 0
    assert opened_urls == []
