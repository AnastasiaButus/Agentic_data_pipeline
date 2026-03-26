"""Domain models and integration contracts for the pipeline."""

from .contracts import (
    APIClientProtocol,
    ArtifactRegistryProtocol,
    DatasetLoaderProtocol,
    LLMClientProtocol,
)
from .dataclasses import SourceCandidate
from .metrics_models import ComparisonMetric, TrainingMetrics
from .report_models import ALIterationResult, AnnotationSpec, LabelResult, QualityReport

__all__ = [
    "APIClientProtocol",
    "ALIterationResult",
    "AnnotationSpec",
    "ArtifactRegistryProtocol",
    "ComparisonMetric",
    "DatasetLoaderProtocol",
    "LLMClientProtocol",
    "LabelResult",
    "QualityReport",
    "SourceCandidate",
    "TrainingMetrics",
]
