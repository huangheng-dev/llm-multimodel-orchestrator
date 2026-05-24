"""Quickstart: fan out a single prompt to 3 LLM providers in parallel.

Set the relevant API keys as env vars before running:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...

Then:
    python examples/quickstart.py
"""
import asyncio
import os

from llm_orchestrator import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    Orchestrator,
)


async def main() -> None:
    # Each provider is configured independently. Skip any you don't have a key for.
    providers = []
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-sonnet-4-6",
        ))
    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o",
        ))
    if os.getenv("GEMINI_API_KEY"):
        providers.append(GeminiProvider(
            api_key=os.environ["GEMINI_API_KEY"],
            model="gemini-1.5-pro",
        ))

    if not providers:
        raise SystemExit(
            "No API keys found. Set at least one of "
            "ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY."
        )

    orch = Orchestrator(providers=providers)
    result = await orch.run("Explain quantum entanglement in one sentence.")

    print(f"\n{result.success_count}/{result.total_count} providers succeeded\n")
    for name, r in result.results.items():
        print(f"=== {name} ({r.attempts} attempt(s), {r.elapsed_ms}ms) ===")
        if r.ok:
            print(r.content)
        else:
            print(f"ERROR: {r.error}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
