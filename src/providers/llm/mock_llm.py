"""Deterministic mock LLM used for tests and offline behavior."""

from __future__ import annotations

from typing import Any

from src.domain import LabelResult

from .base_llm import BaseLLM


class MockLLM(BaseLLM):
    """Return predictable responses using a tiny keyword-based rule set."""

    def generate(self, prompt: str) -> str:
        """Generate a deterministic label-like response from the prompt."""

        text = prompt.lower()
        if "side effect" in text:
            return "side_effects"
        if "energy" in text:
            return "energy"
        return "other"

    def classify_effect(self, text: str, labels: list[str]) -> LabelResult:
        """Classify review text into one of the supported effect labels."""

        if not labels:
            raise ValueError("labels must not be empty")

        text_lower = text.lower()
        if "side effect" in text_lower:
            label = self._choose_label("side_effects", labels)
            confidence = 0.9
        elif "energy" in text_lower:
            label = self._choose_label("energy", labels)
            confidence = 0.9
        else:
            label = self._choose_label("other", labels)
            confidence = 0.55

        return LabelResult(label=label, confidence=confidence)

    def _choose_label(self, predicted: str, labels: list[str]) -> str:
        """Map the predicted label to the provided label vocabulary when possible."""

        normalized_prediction = self._normalize(predicted)
        for label in labels:
            if self._normalize(label) == normalized_prediction:
                return label
        return predicted

    def _normalize(self, label: str) -> str:
        """Normalize label text so space and hyphen variants compare consistently."""

        return label.strip().lower().replace(" ", "_").replace("-", "_")
