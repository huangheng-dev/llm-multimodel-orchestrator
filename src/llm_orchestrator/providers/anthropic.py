"""Anthropic (Claude) Messages API adapter."""
from __future__ import annotations

from typing import Any

from ..types import Image
from .base import Provider, RequestPayload


# Anthropic-specific quirks worth knowing:
#   - Header `anthropic-version` is mandatory; missing it returns 400.
#   - Image source uses {type: 'url'|'base64', ...} — NOT the OpenAI image_url shape.
#   - Image blocks must come BEFORE the text block in the same message for best results.
class AnthropicProvider(Provider):
    name = "anthropic"
    DEFAULT_URL = "https://api.anthropic.com/v1/messages"

    def build_payload(self, prompt: str, images: list[Image]) -> RequestPayload:
        url = self.base_url or self.DEFAULT_URL
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **self.extra_headers,
        }

        content_blocks: list[dict[str, Any]] = []
        for img in images:
            if img.base64:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime,
                        "data": img.base64,
                    },
                })
            elif img.url:
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": img.url},
                })
        content_blocks.append({"type": "text", "text": prompt})

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content_blocks}],
        }
        return RequestPayload(url=url, headers=headers, json_body=body)

    def parse_response(self, body: dict[str, Any]) -> str:
        # Anthropic returns content as a list of blocks; concatenate text blocks.
        content = body.get("content") or []
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
