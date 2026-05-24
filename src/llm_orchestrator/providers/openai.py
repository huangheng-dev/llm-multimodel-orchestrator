"""OpenAI Chat Completions adapter.

Compatible with any OpenAI-protocol gateway: vLLM, Azure OpenAI, OpenRouter,
Aliyun DashScope (compatible mode), LiteLLM, etc. Set `base_url` to override.
"""
from __future__ import annotations

from typing import Any

from ..types import Image
from .base import Provider, RequestPayload


class OpenAIProvider(Provider):
    name = "openai"
    DEFAULT_URL = "https://api.openai.com/v1/chat/completions"

    def build_payload(self, prompt: str, images: list[Image]) -> RequestPayload:
        url = self.base_url or self.DEFAULT_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        # OpenAI format: content is a list of text + image_url parts.
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if img.url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": img.url},
                })
            elif img.base64:
                # Inline base64 via data URI.
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime};base64,{img.base64}"},
                })

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content_parts}],
        }
        return RequestPayload(url=url, headers=headers, json_body=body)

    def parse_response(self, body: dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        # `content` is usually a string. With multimodal / responses API
        # returning structured output it can also be a list of parts.
        # 兼容两种情况 — 见过不少 OpenAI-protocol 兼容网关返回 list 形式。
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def extract_usage(body: dict[str, Any]) -> dict[str, int]:
        """Pull token usage from an OpenAI response (returns zeros if absent).

        Not all gateways forward `usage`. For accurate billing, prefer the
        provider's own dashboard.
        """
        usage = body.get("usage") or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
