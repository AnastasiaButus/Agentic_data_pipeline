"""Minimal Gemini client for Gemini Developer API annotation requests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .base_llm import BaseLLM


GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
ANNOTATION_SENTIMENT_LABELS = ["negative", "neutral", "positive"]
DEFAULT_EFFECT_LABELS = ["energy", "side_effects", "other"]


@dataclass(slots=True)
class GeminiClient(BaseLLM):
    """Call Gemini Developer API for the annotation prompt contract only."""

    api_key: str | None = None
    model_name: str = DEFAULT_GEMINI_MODEL
    timeout_seconds: float = 30.0
    opener: Callable[..., Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Bind a default transport while keeping the client easy to mock in tests."""

        if self.opener is None:
            self.opener = urlopen

    def generate(self, prompt: str) -> str:
        """Generate structured annotation output through the Gemini REST API."""

        if not self.api_key:
            raise ValueError("Gemini API key is required")

        request_body = self.build_request_payload(prompt)
        response_body = self._post_json(request_body)
        return self._extract_text(response_body)

    def generate_with_schema(self, prompt: str, response_schema: dict[str, Any]) -> str:
        """Generate structured JSON output with a caller-provided response schema."""

        if not self.api_key:
            raise ValueError("Gemini API key is required")

        request_body = self.build_request_payload(prompt, response_schema=response_schema)
        response_body = self._post_json(request_body)
        return self._extract_text(response_body)

    def build_request_payload(self, prompt: str, response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build the JSON-first Gemini request body for the annotation contract."""

        schema = response_schema
        if schema is None:
            effect_labels = self._extract_effect_labels(prompt)
            schema = self._build_response_schema(effect_labels)
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }

    def build_request_url(self) -> str:
        """Build the REST endpoint URL for a Gemini text generation request."""

        model_name = quote(self.model_name or DEFAULT_GEMINI_MODEL, safe="-._~")
        return f"{GEMINI_API_BASE_URL}/models/{model_name}:generateContent"

    def _post_json(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON request and parse the JSON response using the injected transport."""

        request = Request(
            self.build_request_url(),
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-goog-api-key": self.api_key or "",
            },
            method="POST",
        )

        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                response_bytes = response.read()
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Gemini response was not valid JSON") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Gemini response payload must be a JSON object")

        return payload

    def _extract_text(self, response_payload: dict[str, Any]) -> str:
        """Extract the model text from the Gemini response payload."""

        if not response_payload:
            raise RuntimeError("Gemini response payload is empty")

        candidates = response_payload.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if text is not None:
                        return str(text)

        text = response_payload.get("text")
        if text is not None:
            return str(text)

        raise RuntimeError("Gemini response did not include text content")

    def _build_response_schema(self, effect_labels: list[str]) -> dict[str, Any]:
        """Build the JSON schema for the annotation contract."""

        effect_enum = effect_labels or list(DEFAULT_EFFECT_LABELS)
        return {
            "type": "object",
            "additionalProperties": False,
            "propertyOrdering": ["effect_label", "sentiment_label", "confidence"],
            "properties": {
                "effect_label": {
                    "type": "string",
                    "enum": effect_enum,
                    "description": "Effect label for the review text.",
                },
                "sentiment_label": {
                    "type": "string",
                    "enum": list(ANNOTATION_SENTIMENT_LABELS),
                    "description": "Sentiment label derived from the review.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score in the closed interval [0, 1].",
                },
            },
            "required": ["effect_label", "sentiment_label", "confidence"],
        }

    def _extract_effect_labels(self, prompt: str) -> list[str]:
        """Extract the effect-label vocabulary from the existing annotation prompt contract."""

        patterns = [
            r"^\s*Допустимые\s+effect_label:\s*(.+?)\s*$",
            r"^\s*(?:Допустимые|Разрешенные|Разрешённые|Allowed)\s+effect_label(?:s)?\s*[:=]\s*(.+?)\s*$",
            r"^\s*effect_label(?:s)?\s*(?:allowed|options|labels)?\s*[:=]\s*(.+?)\s*$",
        ]

        for line in prompt.splitlines():
            for pattern in patterns:
                match = re.match(pattern, line, flags=re.IGNORECASE)
                if not match:
                    continue
                cleaned = self._parse_effect_label_candidates(match.group(1))
                if cleaned:
                    return cleaned
        return list(DEFAULT_EFFECT_LABELS)

    def _parse_effect_label_candidates(self, raw_labels: Any) -> list[str]:
        """Parse and sanitize prompt-provided effect-label candidates."""

        normalized_raw = str(raw_labels).strip().strip("[]")
        labels = [self._normalize_label(label) for label in normalized_raw.split(",")]
        cleaned: list[str] = []
        for label in labels:
            if not label:
                continue
            if not re.match(r"^[a-z0-9_]+$", label):
                continue
            cleaned.append(label)
        return cleaned

    def _normalize_label(self, label: Any) -> str:
        """Normalize label text so prompt extraction stays stable."""

        return str(label).strip().lower().replace(" ", "_").replace("-", "_")
