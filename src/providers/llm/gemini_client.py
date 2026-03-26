"""Minimal Gemini client stub used for dependency injection and tests."""

from __future__ import annotations

from dataclasses import dataclass

from .base_llm import BaseLLM


@dataclass(slots=True)
class GeminiClient(BaseLLM):
    """Hold Gemini configuration while leaving generation for a future step."""

    api_key: str | None = None
    model_name: str = "gemini-2.5-flash"

    def generate(self, prompt: str) -> str:
        """Generation is intentionally not implemented in this step."""

        raise NotImplementedError("GeminiClient.generate is not implemented yet")
