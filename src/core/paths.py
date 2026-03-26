"""Filesystem paths used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelinePaths:
    """Derive the standard directory layout from a single project root."""

    root_dir: Path
    data_raw: Path = field(init=False)
    data_interim: Path = field(init=False)
    data_labeled: Path = field(init=False)
    models: Path = field(init=False)
    reports: Path = field(init=False)
    reports_figures: Path = field(init=False)
    configs: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.data_raw = self.root_dir / "data" / "raw"
        self.data_interim = self.root_dir / "data" / "interim"
        self.data_labeled = self.root_dir / "data" / "labeled"
        self.models = self.root_dir / "models"
        self.reports = self.root_dir / "reports"
        self.reports_figures = self.reports / "figures"
        self.configs = self.root_dir / "configs"

    def ensure_dirs(self) -> None:
        """Create the standard directory tree and keep the call idempotent."""

        for path in (
            self.data_raw,
            self.data_interim,
            self.data_labeled,
            self.models,
            self.reports,
            self.reports_figures,
            self.configs,
        ):
            path.mkdir(parents=True, exist_ok=True)