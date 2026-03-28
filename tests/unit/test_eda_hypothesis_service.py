"""Unit tests for LLM-assisted and heuristic EDA hypothesis generation."""

from __future__ import annotations

from pathlib import Path

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
from src.core.context import PipelineContext
from src.services.eda_hypothesis_service import EDAHypothesisService


def _make_context(
    tmp_path: Path,
    *,
    use_llm: bool = True,
    llm_provider: str = "mock",
) -> PipelineContext:
    """Build a minimal context for EDA hypothesis tests."""

    config = AppConfig(
        project=ProjectConfig(name="eda-hypothesis-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(use_llm=use_llm, llm_provider=llm_provider),
        request=RequestConfig(topic="fitness supplements", domain="supplements", modality="text"),
    )
    return PipelineContext.from_config(config)


def _eda_context() -> dict[str, object]:
    """Return a compact but realistic EDA context payload."""

    return {
        "n_rows": 8,
        "duplicate_summary": {"available": True, "duplicate_rows": 2},
        "raw_vs_cleaned": {"available": True, "raw_rows": 10, "cleaned_rows": 8, "dropped_rows": 2},
        "missing_values_summary": {
            "text": {"available": True, "missing_count": 0},
            "source": {"available": True, "missing_count": 1},
        },
        "effect_label_distribution": {
            "available": True,
            "counts": {"energy": 6, "side_effects": 1, "other": 1},
        },
        "cleaned_word_cloud": {
            "available": True,
            "valid_text_rows": 8,
            "token_count": 20,
            "unique_terms": 10,
            "terms": [
                {"term": "energy", "count": 5},
                {"term": "boost", "count": 3},
                {"term": "focus", "count": 2},
            ],
        },
        "quality_warnings": ["duplicates removed"],
        "notes": ["dataset remains topic-focused"],
    }


def test_eda_hypothesis_service_builds_heuristic_summary_when_llm_disabled(tmp_path: Path) -> None:
    """Heuristic mode should still produce operator-facing hypotheses and follow-ups."""

    service = EDAHypothesisService(_make_context(tmp_path, use_llm=False), llm_client=None)

    summary = service.build_summary(_eda_context())

    assert summary["available"] is True
    assert summary["hypothesis_mode"] == "llm_disabled_heuristic_only"
    assert summary["provider_status"] == "disabled_in_config"
    assert summary["n_hypotheses"] >= 1
    assert summary["hitl_followups"]
    assert any(item["priority"] == "high" for item in summary["hypotheses"])


def test_eda_hypothesis_service_falls_back_when_gemini_requested_but_unavailable(tmp_path: Path) -> None:
    """Gemini-requested runs should degrade safely to heuristics when no structured client is active."""

    class MockLLMClient:
        pass

    service = EDAHypothesisService(
        _make_context(tmp_path, use_llm=True, llm_provider="gemini"),
        llm_client=MockLLMClient(),
    )

    summary = service.build_summary(_eda_context())

    assert summary["hypothesis_mode"] == "gemini_requested_but_unavailable_heuristic_fallback"
    assert summary["requested_provider"] == "gemini"
    assert summary["resolved_provider"] == "mockllmclient"
    assert summary["provider_status"] == "gemini_requested_but_unavailable_heuristic_fallback"
    assert summary["n_hypotheses"] >= 1


def test_eda_hypothesis_service_uses_structured_gemini_output_when_available(tmp_path: Path) -> None:
    """A Gemini-like client with structured output should populate graph-grounded hypotheses."""

    captured: dict[str, object] = {}

    def _generate_with_schema(self, prompt: str, response_schema: dict[str, object]) -> str:
        captured["prompt"] = prompt
        captured["schema"] = response_schema
        return """
        {
          "overall_note": "EDA signals suggest a narrow but reviewable batch.",
          "hypotheses": [
            {
              "title": "Label concentration",
              "observation": "energy dominates the current cleaned set",
              "hypothesis": "source mix is currently too narrow",
              "hitl_action": "review source approval before retrain",
              "priority": "high"
            }
          ]
        }
        """

    FakeGeminiClient = type("GeminiClient", (), {"generate_with_schema": _generate_with_schema})

    service = EDAHypothesisService(
        _make_context(tmp_path, use_llm=True, llm_provider="gemini"),
        llm_client=FakeGeminiClient(),
    )

    summary = service.build_summary(_eda_context())

    assert summary["hypothesis_mode"] == "llm_generate_parse"
    assert summary["requested_provider"] == "gemini"
    assert summary["resolved_provider"] == "gemini"
    assert summary["provider_status"] == "gemini_active_for_eda_hypotheses"
    assert summary["overall_note"] == "EDA signals suggest a narrow but reviewable batch."
    assert summary["n_hypotheses"] == 1
    assert summary["hypotheses"][0]["title"] == "Label concentration"
    assert summary["hitl_followups"] == ["review source approval before retrain"]
    assert "EDA summary" in str(captured["prompt"])
    assert captured["schema"]["type"] == "object"


def test_eda_hypothesis_service_reports_missing_context_explicitly(tmp_path: Path) -> None:
    """Missing EDA context should be explained instead of silently producing fake notes."""

    service = EDAHypothesisService(_make_context(tmp_path), llm_client=None)

    summary = service.build_summary({})

    assert summary["available"] is False
    assert summary["hypothesis_mode"] == "not_available_no_eda_context"
    assert summary["provider_status"] == "not_available_no_eda_context"
    assert summary["n_hypotheses"] == 0
