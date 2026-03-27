"""Unit tests for the Gemini annotation provider and fallback behavior."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest

from src.agents.annotation_agent import AnnotationAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
from src.core.context import PipelineContext
from src.providers.llm.gemini_client import GeminiClient
from src.providers.llm.mock_llm import MockLLM
from src.services.pipeline_controller import PipelineController


class FakeRegistry:
    """Capture artifact writes without touching the filesystem."""

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        return Path(path)


class _Frame:
    """Tiny dataframe-like helper used to keep the test independent of pandas."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    @property
    def empty(self) -> bool:
        return not self._rows

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        if orient != "records":
            raise ValueError("Only records orientation is supported")
        return [dict(row) for row in self._rows]


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for Gemini annotation tests."""

    config = AppConfig(
        project=ProjectConfig(name="gemini-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(confidence_threshold=0.6, use_llm=True),
        request=RequestConfig(topic="fitness supplements", domain="supplements", modality="text"),
    )
    return PipelineContext.from_config(config)


def test_gemini_client_builds_json_first_request_and_extracts_text() -> None:
    """Gemini client should build the structured-output request and map the response text."""

    captured: dict[str, object] = {}

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"effect_label": "energy", "sentiment_label": "positive", "confidence": 0.9}'
                                }
                            ]
                        }
                    }
                ]
            }
        )

    client = GeminiClient(api_key="test-key", opener=fake_opener)

    prompt = "\n".join(
        [
            "Ты разметчик отзывов о пищевых добавках.",
            "Верни только JSON без пояснений, markdown и лишнего текста.",
            "",
            "Допустимые effect_label: energy, side_effects, other",
            "Текст отзыва:",
            "Пример отзыва",
        ]
    )

    response_text = client.generate(prompt)

    assert response_text == '{"effect_label": "energy", "sentiment_label": "positive", "confidence": 0.9}'
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert captured["headers"]["X-goog-api-key"] == "test-key"
    assert captured["body"]["contents"][0]["parts"][0]["text"] == prompt
    schema = captured["body"]["generationConfig"]["responseJsonSchema"]
    assert schema["properties"]["effect_label"]["enum"] == ["energy", "side_effects", "other"]
    assert schema["properties"]["confidence"]["minimum"] == 0
    assert schema["properties"]["confidence"]["maximum"] == 1
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_pipeline_controller_selects_annotation_provider_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Controller selection should follow llm_provider, not environment-only heuristics."""

    context = _make_context(tmp_path)

    context.config.annotation.use_llm = False
    controller = PipelineController(context)
    assert controller.annotation_agent.llm_client is None

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    context.config.annotation.llm_provider = "mock"
    context.config.annotation.use_llm = True
    controller = PipelineController(context)
    assert isinstance(controller.annotation_agent.llm_client, MockLLM)

    context.config.annotation.llm_provider = "gemini"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = PipelineController(context)
    assert isinstance(controller.annotation_agent.llm_client, MockLLM)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    controller = PipelineController(context)
    assert isinstance(controller.annotation_agent.llm_client, GeminiClient)

    context.config.annotation.llm_provider = "anything-else"
    controller = PipelineController(context)
    assert isinstance(controller.annotation_agent.llm_client, MockLLM)


def test_annotation_agent_falls_back_when_gemini_provider_raises(tmp_path: Path) -> None:
    """A Gemini provider exception should fall back to the deterministic offline path."""

    class BrokenGeminiClient(GeminiClient):
        def generate(self, prompt: str) -> str:
            raise RuntimeError("boom")

    agent = AnnotationAgent(_make_context(tmp_path), llm_client=BrokenGeminiClient(api_key="test-key"), registry=FakeRegistry())
    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "Average result", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    rows = labeled.to_dict(orient="records")
    trace = agent.get_annotation_trace()

    assert rows[0]["effect_label"] == "other"
    assert rows[0]["sentiment_label"] == "positive"
    assert rows[0]["confidence"] == 0.5
    assert trace["llm_mode"] == "generate_parse"
    assert trace["n_fallback_rows"] == 1
    assert trace["fallback_rows"][0]["mode"] == "generate_error"


def test_annotation_agent_parses_gemini_json_response(tmp_path: Path) -> None:
    """Gemini JSON responses should be parsed structurally without falling back."""

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def fake_opener(request, timeout):
        return FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"effect_label": "energy", "sentiment_label": "positive", "confidence": 0.88}'
                                }
                            ]
                        }
                    }
                ]
            }
        )

    client = GeminiClient(api_key="test-key", opener=fake_opener)
    agent = AnnotationAgent(_make_context(tmp_path), llm_client=client, registry=FakeRegistry())

    labeled = agent.auto_label(
        _Frame(
            [
                {"id": "1", "source": "HF", "text": "This supplement gives me more energy.", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"},
            ]
        )
    )

    rows = labeled.to_dict(orient="records")
    trace = agent.get_annotation_trace()

    assert rows[0]["effect_label"] == "energy"
    assert rows[0]["sentiment_label"] == "positive"
    assert rows[0]["confidence"] == 0.88
    assert trace["llm_mode"] == "generate_parse"
    assert trace["n_fallback_rows"] == 0
    assert trace["parser_contract"]["parse_status_counts"]["parsed"] == 1


def test_gemini_client_raises_runtime_error_on_transport_failure() -> None:
    """Transport failures should surface as runtime errors for the caller to handle."""

    def broken_opener(request, timeout):
        raise URLError("network down")

    client = GeminiClient(api_key="test-key", opener=broken_opener)

    with pytest.raises(RuntimeError, match="Gemini request failed"):
        client.generate("Ты разметчик отзывов о пищевых добавках.\nДопустимые effect_label: energy")


def test_gemini_client_extract_effect_labels_old_prompt_marker() -> None:
    """The legacy prompt marker should keep extracting effect labels as before."""

    client = GeminiClient(api_key="test-key", opener=lambda *_args, **_kwargs: None)

    labels = client._extract_effect_labels("Допустимые effect_label: energy, side_effects, other")

    assert labels == ["energy", "side_effects", "other"]


def test_gemini_client_extract_effect_labels_with_changed_wording() -> None:
    """Slightly changed prompt wording should still produce the effect-label vocabulary."""

    client = GeminiClient(api_key="test-key", opener=lambda *_args, **_kwargs: None)

    prompt = "Allowed effect_labels = energy, side effects, other"
    labels = client._extract_effect_labels(prompt)

    assert labels == ["energy", "side_effects", "other"]


def test_gemini_client_extract_effect_labels_falls_back_safely_when_not_parseable() -> None:
    """If labels cannot be extracted from prompt text, defaults should be used safely."""

    client = GeminiClient(api_key="test-key", opener=lambda *_args, **_kwargs: None)

    prompt = "effect_label: одно из значений списка ниже"
    labels = client._extract_effect_labels(prompt)

    assert labels == ["energy", "side_effects", "other"]
