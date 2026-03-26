"""Tests for the shared base agent behavior."""

from __future__ import annotations

from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.services.artifact_registry import ArtifactRegistry


class ExampleAgent(BaseAgent):
    """Concrete test agent used to validate the base-class contract."""


def test_base_agent_name_matches_class_name(tmp_path: Path) -> None:
    """The base agent should expose the concrete class name for tracing."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    context = PipelineContext.from_config(config)
    registry = ArtifactRegistry(context)

    agent = ExampleAgent(context, registry)

    assert agent.name == "ExampleAgent"
    assert agent.ctx == context
    assert agent.registry == registry
