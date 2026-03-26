"""Domain models that capture quality, annotation, and active-learning reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QualityReport:
    """Summarize basic data-quality signals for a dataset or batch."""

    missing: dict[str, float] = field(default_factory=dict)
    duplicates: int = 0
    outliers: dict[str, object] = field(default_factory=dict)
    imbalance: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary for serialization or logging."""

        return asdict(self)


@dataclass(slots=True)
class AnnotationSpec:
    """Describe the annotation contract in a structured, markdown-friendly form."""

    name: str
    description: str
    text_field: str = "text"
    label_field: str = "label"
    id_field: str = "id"
    labels: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    output_format: str = "markdown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary for spec rendering or persistence."""

        return asdict(self)


@dataclass(slots=True)
class LabelResult:
    """Capture a single label decision and the confidence behind it."""

    label: str
    confidence: float
    rationale: str | None = None

    def __post_init__(self) -> None:
        """Validate the confidence boundary so edge values remain accepted."""

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0 inclusive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary for downstream consumers."""

        return asdict(self)


@dataclass(slots=True)
class ALIterationResult:
    """Summarize one active-learning iteration in a compact, serializable form."""

    iteration: int
    n_labeled: int
    accuracy: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        """Return the iteration summary as a plain dictionary."""

        return asdict(self)
