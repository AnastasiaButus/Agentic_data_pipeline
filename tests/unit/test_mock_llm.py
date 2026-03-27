"""Tests for the deterministic mock LLM provider."""

from __future__ import annotations

import pytest

from src.domain import LabelResult
from src.providers.llm.mock_llm import MockLLM


@pytest.mark.parametrize(
    "text, expected",
    [
        ("This supplement gives me more energy.", "energy"),
        ("I noticed a side effect after a week.", "side_effects"),
        ("Nothing special here.", "other"),
    ],
)
def test_mock_llm_classifies_effects(text: str, expected: str) -> None:
    """The classifier should map key phrases to deterministic labels."""

    llm = MockLLM()

    result = llm.classify_effect(text, labels=["energy", "side_effects", "other"])

    assert isinstance(result, LabelResult)
    assert result.label == expected
    assert 0.0 <= result.confidence <= 1.0


def test_mock_llm_returns_label_result_and_bounds_confidence() -> None:
    """The classifier should always return a LabelResult with boundary-safe confidence."""

    llm = MockLLM()

    result = llm.classify_effect("energy boost", labels=["energy", "other"])

    assert isinstance(result, LabelResult)
    assert result.confidence == 0.9


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Crafting instructions for a new build", "crafting"),
        ("Combat tips for the arena", "combat"),
        ("Enchantments help long-term progression", "enchantments"),
        ("Nothing specific here", "other"),
    ],
)
def test_mock_llm_classifies_minecraft_labels(text: str, expected: str) -> None:
    """The classifier should respect minecraft-oriented label vocabularies."""

    llm = MockLLM()

    result = llm.classify_effect(text, labels=["crafting", "combat", "enchantments", "other"])

    assert result.label == expected
    assert result.confidence in {0.55, 0.9}


def test_mock_llm_empty_labels_raise_value_error() -> None:
    """An empty label vocabulary should fail fast."""

    llm = MockLLM()

    with pytest.raises(ValueError):
        llm.classify_effect("energy boost", labels=[])
