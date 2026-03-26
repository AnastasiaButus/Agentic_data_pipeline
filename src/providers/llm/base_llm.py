"""Shared base class for LLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Define the minimal generation contract used by provider adapters."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text completion for the provided prompt."""
