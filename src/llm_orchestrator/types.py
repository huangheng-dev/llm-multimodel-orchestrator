"""Data models for the orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ResultStatus(str, Enum):
    """Outcome of a single provider call."""
    OK = "ok"
    FAILED = "failed"


@dataclass
class Image:
    """An input image for multimodal prompts.

    Either supply `base64` (raw base64 string without data: prefix) or `url`.
    `mime` defaults to image/png and is required by Anthropic / Gemini.
    """
    base64: Optional[str] = None
    url: Optional[str] = None
    mime: str = "image/png"

    def __post_init__(self) -> None:
        if not self.base64 and not self.url:
            raise ValueError("Image needs either base64 or url")


@dataclass
class ProviderResult:
    """Outcome of one provider call (success or failure, plus diagnostics)."""
    provider: str
    status: ResultStatus
    content: str = ""
    error: str = ""
    attempts: int = 1
    elapsed_ms: int = 0
    raw: Optional[dict[str, Any]] = None  # original response body (for debugging)

    @property
    def ok(self) -> bool:
        return self.status == ResultStatus.OK


@dataclass
class OrchestrationResult:
    """The full picture after all providers have been queried."""
    results: dict[str, ProviderResult] = field(default_factory=dict)
    consensus: Optional[Any] = None  # type depends on consensus strategy
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results.values() if r.ok)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> bool:
        """At least one provider succeeded."""
        return self.success_count > 0

    def by_provider(self, provider: str) -> Optional[ProviderResult]:
        return self.results.get(provider)
