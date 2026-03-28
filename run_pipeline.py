"""Command-line entry point for the universal agentic data pipeline."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Any

from src.core.config import load_config
from src.core.context import PipelineContext
from src.services.pipeline_controller import PipelineController


def _resolve_report_path(ctx: PipelineContext, result: dict[str, Any], report_key: str) -> Path | None:
    """Resolve a report path from the compact controller summary into an absolute filesystem path."""

    reports = result.get("reports", {})
    if not isinstance(reports, dict):
        return None

    relative_path = reports.get(report_key)
    if not relative_path:
        return None

    return (ctx.paths.root_dir / Path(str(relative_path))).resolve()


def _print_run_summary(ctx: PipelineContext, result: dict[str, Any]) -> None:
    """Print a short operator-facing summary with the main artifact paths."""

    dashboard_path = _resolve_report_path(ctx, result, "dashboard")
    final_report_path = _resolve_report_path(ctx, result, "final_report")
    eda_html_path = _resolve_report_path(ctx, result, "eda_html_report")
    review_workspace_path = _resolve_report_path(ctx, result, "review_workspace")

    print("Pipeline run completed.")
    print(f"Runtime mode: {result.get('runtime_mode', 'unknown')}")
    if dashboard_path is not None:
        print(f"Dashboard: {dashboard_path}")
    if final_report_path is not None:
        print(f"Final report: {final_report_path}")
    if eda_html_path is not None:
        print(f"EDA HTML: {eda_html_path}")
    if review_workspace_path is not None:
        print(f"Review workspace: {review_workspace_path}")

    review_status = str(result.get("review_status", "") or "")
    if review_status == "skipped_missing_corrected_queue":
        print("Next action: if review is needed, fill review_queue_corrected.csv and rerun the pipeline.")


def _open_artifact(path: Path | None) -> None:
    """Open a generated local artifact in the default browser when it exists."""

    if path is None or not path.exists():
        return
    webbrowser.open(path.as_uri())


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, load the config, and execute the pipeline."""

    parser = argparse.ArgumentParser(description="Run the universal agentic data pipeline demo")
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config")
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Open reports/run_dashboard.html after a successful run",
    )
    parser.add_argument(
        "--open-review-workspace",
        action="store_true",
        help="Open reports/review_workspace.html after a successful run",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    ctx = PipelineContext.from_config(config)
    controller = PipelineController(ctx)
    result = controller.run()
    _print_run_summary(ctx, result)

    if args.open_dashboard:
        _open_artifact(_resolve_report_path(ctx, result, "dashboard"))
    if args.open_review_workspace:
        _open_artifact(_resolve_report_path(ctx, result, "review_workspace"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
