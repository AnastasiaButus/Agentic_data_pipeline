"""Minimal base class shared by pipeline agents."""

from __future__ import annotations

import logging

from src.core.context import PipelineContext
from src.services.artifact_registry import ArtifactRegistry


class BaseAgent:
    """Provide shared context, artifact access, and logging for agents."""

    def __init__(self, ctx: PipelineContext, registry: ArtifactRegistry) -> None:
        """Store the execution context and the shared artifact registry."""

        self.ctx = ctx
        self.registry = registry
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def name(self) -> str:
        """Return the concrete class name for logging and tracing."""

        return self.__class__.__name__
