"""Google Gemini (generateContent) adapter.

Gemini's API shape is different from OpenAI/Anthropic in two annoying ways:
    1. Auth is via URL query param (`?key=...`), not a header.
    2. Model name is part of the path (`models/gemini-1.5-pro:generateContent`).
       Typos in the model name return 404, not a clean error message.
"""
from __future__ import annotations

from typing import Any

from ..types import Image
from .base import Provider, RequestPayload


class GeminiProvider(Provider):
    name = "gemini"
    DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def build_payload(self, prompt: str, images: list[Image]) -> RequestPayload:
        base = self.base_url or self.DEFAULT_URL
        url = f"{base.rstrip('/')}/{self.model}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        parts: list[dict[str, Any]] = []
        for img in images:
            # Gemini doesn't accept image URLs — only inline base64. If you
            # only have a URL, fetch the bytes yourself and pass base64.
            if img.url and not img.base64:
                raise ValueError(
                    f"{self.name}: image URLs are not supported; pass base64 bytes."
                )
            parts.append({
                "inline_data": {"mime_type": img.mime, "data": img.base64},
            })
        parts.append({"text": prompt})

        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        return RequestPayload(url=url, headers=headers, json_body=body)

    def parse_response(self, body: dict[str, Any]) -> str:
        # Gemini wraps the answer deep: candidates[0].content.parts[].text
        # Empty `candidates` usually means the model refused (safety filter);
        # we leave detection to the caller via `ProviderResult.raw`.
        candidates = body.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
