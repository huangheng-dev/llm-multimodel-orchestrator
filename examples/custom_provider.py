"""Roll your own provider — point at any OpenAI-compatible gateway.

Many internal LLM gateways (LiteLLM, OpenRouter, vLLM, Aliyun DashScope
compatible mode, etc.) speak the OpenAI Chat Completions protocol. You can
either reuse OpenAIProvider with a custom base_url, or subclass Provider for
a completely custom API shape.
"""
import asyncio
import os
from typing import Any

from llm_orchestrator import (
    OpenAIProvider,
    Orchestrator,
    Provider,
    RequestPayload,
)
from llm_orchestrator.types import Image


# --- Option A: reuse OpenAIProvider with a different base_url ---

def make_litellm_provider():
    """LiteLLM proxy speaks OpenAI protocol."""
    return OpenAIProvider(
        api_key=os.environ["LITELLM_API_KEY"],
        model="claude-3-5-sonnet",   # whatever model LiteLLM routes to
        base_url="http://localhost:4000/v1/chat/completions",
    )


# --- Option B: implement a fully custom provider ---

class OllamaProvider(Provider):
    """Local Ollama: http://localhost:11434/api/chat — text only."""

    name = "ollama"
    DEFAULT_URL = "http://localhost:11434/api/chat"

    def build_payload(self, prompt: str, images: list[Image]) -> RequestPayload:
        if images:
            raise NotImplementedError("This minimal Ollama adapter is text-only.")
        return RequestPayload(
            url=self.base_url or self.DEFAULT_URL,
            headers={"Content-Type": "application/json"},
            json_body={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.temperature},
            },
        )

    def parse_response(self, body: dict[str, Any]) -> str:
        return (body.get("message") or {}).get("content", "").strip()


async def main() -> None:
    providers = [
        OllamaProvider(api_key="ignored", model="llama3.2"),
    ]
    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
        ))

    orch = Orchestrator(providers=providers)
    result = await orch.run("In one sentence, what is Newton's first law?")

    for name, r in result.results.items():
        print(f"{name}: {r.content if r.ok else f'[failed] {r.error}'}\n")


if __name__ == "__main__":
    asyncio.run(main())
