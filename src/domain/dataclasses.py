"""Core domain entities that describe candidate sources for the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SourceType = Literal["hf_dataset", "api", "github_repo", "scrape"]


@dataclass(slots=True)
class SourceCandidate:
    """Describe a potential source in a transport-agnostic way."""

    source_id: str
    source_type: SourceType
    title: str
    uri: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation of the candidate."""

        return asdict(self)
