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
        if "crafting" in text or "redstone" in text or "recipe" in text:
            return "crafting"
        if "combat" in text or "fight" in text or "battle" in text or "potion" in text:
            return "combat"
        if "enchant" in text or "enchanted" in text or "enchantment" in text:
            return "enchantments"
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
        normalized_labels = {self._normalize(label) for label in labels}

        if {"crafting", "combat", "enchantments"} & normalized_labels:
            keyword_routes = [
                (("enchant", "enchantment", "enchanted", "bookshelf", "anvil"), "enchantments"),
                (("combat", "fight", "battle", "arena", "potion", "warning", "attack"), "combat"),
                (("craft", "crafting", "redstone", "build", "recipe"), "crafting"),
            ]
        else:
            keyword_routes = [
                (("side effect", "side effects"), "side_effects"),
                (("energy",), "energy"),
            ]

        for keywords, predicted_label in keyword_routes:
            if any(keyword in text_lower for keyword in keywords):
                return LabelResult(label=self._resolve_label(predicted_label, labels), confidence=0.9)

        return LabelResult(label=self._fallback_label(labels), confidence=0.55)

    def _resolve_label(self, predicted: str, labels: list[str]) -> str:
        """Map the predicted label to the provided label vocabulary when possible."""

        normalized_prediction = self._normalize(predicted)
        for label in labels:
            if self._normalize(label) == normalized_prediction:
                return label
        return self._fallback_label(labels)

    def _fallback_label(self, labels: list[str]) -> str:
        """Pick the safest fallback label for deterministic offline demo behavior."""

        for label in labels:
            if self._normalize(label) == "other":
                return label
        return labels[0]

    def _normalize(self, label: str) -> str:
        """Normalize label text so space and hyphen variants compare consistently."""

        return label.strip().lower().replace(" ", "_").replace("-", "_")
