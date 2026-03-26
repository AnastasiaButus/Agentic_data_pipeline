"""Metrics models used by the domain layer for comparison and training outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ComparisonMetric:
    """Compare a current value against a baseline value."""

    name: str
    current: float
    baseline: float | None = None
    delta: float | None = None
    higher_is_better: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary for comparison reporting."""

        return asdict(self)


@dataclass(slots=True)
class TrainingMetrics:
    """Collect JSON-serializable training metrics for model experiments."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    loss: float = 0.0
    n_examples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary of the recorded metrics."""

        return asdict(self)
