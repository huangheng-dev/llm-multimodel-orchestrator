"""Unit tests with a fake provider — no real API calls."""
import asyncio
from typing import Any

import pytest

from llm_orchestrator import (
    MajorityVote,
    Orchestrator,
    Provider,
    RequestPayload,
)
from llm_orchestrator.types import Image, OrchestrationResult


class FakeProvider(Provider):
    """A provider that returns a fixed answer, no network involved.

    We bypass the HTTP layer by overriding `build_payload` + `parse_response`
    such that the orchestrator gets the answer directly. For real tests of the
    retry / HTTP layer, mock httpx.AsyncClient via respx.
    """

    def __init__(self, name: str, answer: str, **kwargs):
        super().__init__(api_key="fake", model="fake", **kwargs)
        self._answer = answer
        self.name = name

    def build_payload(self, prompt: str, images: list[Image]) -> RequestPayload:
        # Echo the prompt back so a mock transport can return our `_answer`.
        return RequestPayload(
            url="https://example.invalid/echo",
            headers={"X-Echo": "1"},
            json_body={"echo": self._answer},
        )

    def parse_response(self, body: dict[str, Any]) -> str:
        return body.get("echo", "")


@pytest.mark.asyncio
async def test_duplicate_provider_names_rejected():
    """Two providers with the same name must raise — they would collide as dict keys."""
    p1 = FakeProvider(name="dup", answer="A")
    p2 = FakeProvider(name="dup", answer="B")
    with pytest.raises(ValueError, match="Duplicate provider names"):
        Orchestrator(providers=[p1, p2])


@pytest.mark.asyncio
async def test_empty_providers_rejected():
    with pytest.raises(ValueError, match="At least one provider"):
        Orchestrator(providers=[])


def test_majority_vote_combines():
    """The majority strategy returns the most common normalized answer."""
    from llm_orchestrator.types import ProviderResult, ResultStatus

    successful = [
        ProviderResult(provider="a", status=ResultStatus.OK, content="YES"),
        ProviderResult(provider="b", status=ResultStatus.OK, content="yes "),
        ProviderResult(provider="c", status=ResultStatus.OK, content="NO"),
    ]
    consensus = MajorityVote(min_votes=2).combine(successful)
    assert consensus is not None
    assert consensus["votes"] == 2
    assert "a" in consensus["voters"] and "b" in consensus["voters"]


def test_majority_vote_no_majority_returns_none():
    """When no answer hits min_votes, return None (disagreement)."""
    from llm_orchestrator.types import ProviderResult, ResultStatus

    successful = [
        ProviderResult(provider="a", status=ResultStatus.OK, content="X"),
        ProviderResult(provider="b", status=ResultStatus.OK, content="Y"),
        ProviderResult(provider="c", status=ResultStatus.OK, content="Z"),
    ]
    assert MajorityVote(min_votes=2).combine(successful) is None
