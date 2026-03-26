"""Runtime context that binds configuration to the resolved filesystem layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .paths import PipelinePaths


@dataclass(slots=True)
class PipelineContext:
    """Bundle the active config together with its derived filesystem paths."""

    config: AppConfig
    paths: PipelinePaths

    @classmethod
    def from_config(cls, config: AppConfig, root_dir: Path | None = None) -> "PipelineContext":
        """Create a context from config, optionally overriding the project root."""

        resolved_root = Path(root_dir) if root_dir is not None else Path(config.project.root_dir)
        return cls(config=config, paths=PipelinePaths(resolved_root))