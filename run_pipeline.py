"""Command-line entry point for the demo fitness supplements pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.config import load_config
from src.core.context import PipelineContext
from src.services.pipeline_controller import PipelineController


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, load the config, and execute the pipeline."""

    parser = argparse.ArgumentParser(description="Run the universal agentic data pipeline demo")
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    ctx = PipelineContext.from_config(config)
    controller = PipelineController(ctx)
    controller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())