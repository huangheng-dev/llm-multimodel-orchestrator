"""Retry classification — what's worth retrying, what isn't.

The rule of thumb: anything that smells like the upstream had a bad
millisecond (timeout, 5xx, 429) → retry. Anything that smells like
configuration error (4xx auth, malformed URL) → fail fast, retrying
would just burn quota.
"""
from __future__ import annotations

import asyncio

import httpx


# 408 = Request Timeout, 429 = Too Many Requests. Everything 5xx as well.
# We deliberately exclude 4xx other than 408/429 — repeated 401s won't
# magically become 200s and they cost rate-limit budget.
RETRYABLE_HTTP_STATUS = {408, 429}


def is_retryable_http(status: int) -> bool:
    return status in RETRYABLE_HTTP_STATUS or 500 <= status <= 599


def is_retryable_exception(exc: BaseException) -> bool:
    """Transient network conditions worth a second try."""
    # Note: httpx exception hierarchy is wider than this — we explicitly
    # enumerate the ones we've seen flake in production rather than catching
    # the parent class. SSL handshake failures and DNS resolution errors
    # surface as ConnectError, so those are covered.
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return True
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)):
        return True
    return False


async def linear_backoff_sleep(attempt: int, base_ms: int = 500) -> None:
    """Sleep `base_ms * attempt` ms before the next attempt.

    Linear rather than exponential on purpose: with a small max_retries
    (typically 3) exponential gets harsh fast (0.5s / 1s / 2s / 4s ...) and
    most LLM transient errors clear in under a second anyway.
    """
    await asyncio.sleep(base_ms * attempt / 1000)
