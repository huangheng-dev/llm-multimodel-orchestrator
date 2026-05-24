"""Provider base class.

A provider knows how to:
    1. Build a request body for its target API (multimodal-aware)
    2. Parse the response into plain text
    3. Identify itself (a stable string used as result key)

Built-in providers cover Anthropic / OpenAI / Gemini. To support a new vendor,
subclass `Provider` and implement `build_payload` + `parse_response`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..types import Image


@dataclass
class RequestPayload:
    """Everything needed to fire one HTTP request to a provider."""
    url: str
    method: str = "POST"
    headers: dict[str, str] = None  # type: ignore[assignment]
    json_body: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = {}
        if self.json_body is None:
            self.json_body = {}


class Provider(ABC):
    """Adapter for a single LLM vendor.

    Subclasses must:
        - set `name` (used as the result-dict key)
        - implement `build_payload(prompt, images)` → RequestPayload
        - implement `parse_response(json_body)` → str (plain text answer)

    Subclasses may override `timeout_seconds` and any header/auth handling.
    """

    name: str = "override-me"
    timeout_seconds: float = 60.0

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError(f"{self.name}: api_key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}

    @abstractmethod
    def build_payload(self, prompt: str, images: list[Image]) -> RequestPayload: ...

    @abstractmethod
    def parse_response(self, body: dict[str, Any]) -> str: ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model!r}>"
