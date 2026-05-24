"""Consensus strategies.

A consensus strategy takes the list of successful provider results and
produces a single "agreed" answer (or surfaces disagreement).

Two strategies are built-in:
    - MajorityVote: groups identical answers and picks the largest group
    - FirstSuccess: returns the first successful response (no real voting)

Custom strategies: subclass `ConsensusStrategy` and implement `combine`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, Optional

from .types import ProviderResult


class ConsensusStrategy(ABC):
    """Aggregate multiple provider outcomes into one answer."""

    @abstractmethod
    def combine(self, successful: list[ProviderResult]) -> Optional[Any]:
        """Return the consensus value, or None if no consensus could be reached."""


class FirstSuccess(ConsensusStrategy):
    """Trivial: return whoever answered first / was listed first."""

    def combine(self, successful: list[ProviderResult]) -> Optional[Any]:
        if not successful:
            return None
        # `successful` is already sorted by listed order in the orchestrator.
        return successful[0].content


class MajorityVote(ConsensusStrategy):
    """Group by normalized answer; return the most common one.

    Optionally pass a `normalize` callable to project answers into a comparable
    form. For example, extract a structured signal like "BUY/SELL/HOLD" from
    a long-form analysis and vote on that.

    `min_votes`: minimum number of agreeing providers required to declare
    consensus. Below the threshold, returns None.
    """

    def __init__(
        self,
        normalize: Optional[Callable[[str], str]] = None,
        min_votes: int = 2,
    ) -> None:
        self.normalize = normalize or (lambda s: s.strip().lower())
        self.min_votes = min_votes

    def combine(self, successful: list[ProviderResult]) -> Optional[Any]:
        if not successful:
            return None
        bucketed = [(r, self.normalize(r.content)) for r in successful]
        counts = Counter(key for _, key in bucketed)
        winner_key, votes = counts.most_common(1)[0]
        if votes < self.min_votes:
            return None
        # Return the original content from the first provider in the winning bucket.
        for r, key in bucketed:
            if key == winner_key:
                return {
                    "answer": r.content,
                    "votes": votes,
                    "total": len(successful),
                    "voters": [pr.provider for pr, k in bucketed if k == winner_key],
                }
        return None
