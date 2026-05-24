"""Majority-vote consensus across multiple LLMs.

Use case: high-stakes classification where a single model might hallucinate.
Run the same prompt through 3 providers, normalize each answer to a single
label, and vote.

Example output:
    Consensus: {'answer': 'YES', 'votes': 2, 'total': 3, 'voters': ['anthropic', 'openai']}
"""
import asyncio
import os
import re

from llm_orchestrator import (
    AnthropicProvider,
    GeminiProvider,
    MajorityVote,
    OpenAIProvider,
    Orchestrator,
)


def extract_label(text: str) -> str:
    """Extract a single YES/NO/MAYBE label from a longer answer."""
    m = re.search(r"\b(YES|NO|MAYBE)\b", text.upper())
    return m.group(1) if m else "UNKNOWN"


async def main() -> None:
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

    if len(providers) < 2:
        raise SystemExit("Need at least 2 providers for meaningful voting.")

    orch = Orchestrator(
        providers=providers,
        consensus=MajorityVote(normalize=extract_label, min_votes=2),
    )

    prompt = (
        "Is the following statement scientifically accurate? "
        "Answer with exactly one word: YES, NO, or MAYBE.\n\n"
        "Statement: Water boils at 100 degrees Celsius at sea level."
    )

    result = await orch.run(prompt)

    print("\n--- Individual answers ---")
    for name, r in result.results.items():
        label = extract_label(r.content) if r.ok else "ERROR"
        print(f"{name:12s} → {label}  ({r.attempts} attempt(s))")

    print(f"\n--- Consensus ---")
    if result.consensus:
        print(f"Answer:  {result.consensus['answer'][:80]}...")
        print(f"Votes:   {result.consensus['votes']}/{result.consensus['total']}")
        print(f"Voters:  {', '.join(result.consensus['voters'])}")
    else:
        print("No majority reached — providers disagreed.")


if __name__ == "__main__":
    asyncio.run(main())
