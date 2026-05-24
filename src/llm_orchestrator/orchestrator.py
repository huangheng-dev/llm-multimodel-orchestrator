"""Core: fire N providers in parallel, retry transient failures per-provider."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Sequence

import httpx

from .consensus import ConsensusStrategy
from .providers.base import Provider
from .retry import (
    is_retryable_exception,
    is_retryable_http,
    linear_backoff_sleep,
)
from .types import (
    Image,
    OrchestrationResult,
    ProviderResult,
    ResultStatus,
)

log = logging.getLogger(__name__)


class Orchestrator:
    """Fan out one prompt to multiple LLM providers, gather, optionally vote.

    Each provider retries independently on transient failures (network errors,
    5xx, 408, 429). 4xx auth / validation errors fail fast. Successful providers
    are removed from the retry queue immediately — no duplicate calls.

    Example:
        >>> orch = Orchestrator(providers=[
        ...     AnthropicProvider(api_key=..., model="claude-3-5-sonnet"),
        ...     OpenAIProvider(api_key=..., model="gpt-4o"),
        ... ])
        >>> result = await orch.run("What is 2+2?")
        >>> for name, r in result.results.items():
        ...     print(name, r.content)
    """

    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        max_retries: int = 3,
        retry_backoff_ms: int = 500,
        consensus: Optional[ConsensusStrategy] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider is required")
        # Detect duplicate provider names (would collide as dict keys).
        names = [p.name for p in providers]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate provider names: {names}")

        self.providers = list(providers)
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        self.consensus = consensus
        self._client = client
        self._owns_client = client is None

    async def run(
        self,
        prompt: str,
        images: Optional[list[Image]] = None,
    ) -> OrchestrationResult:
        """Send `prompt` to all providers in parallel and gather their answers."""
        # TODO: streaming variant (yields ProviderResult as each finishes,
        #       useful when you want to render the fastest answer ASAP)
        images = images or []
        client = self._client or httpx.AsyncClient()
        try:
            tasks = [
                self._call_with_retries(client, p, prompt, images)
                for p in self.providers
            ]
            results = await asyncio.gather(*tasks)
        finally:
            if self._owns_client:
                await client.aclose()

        out = OrchestrationResult(
            results={r.provider: r for r in results},
        )
        # Apply consensus over successful results only.
        successful = [r for r in results if r.ok]
        if self.consensus is not None:
            out.consensus = self.consensus.combine(successful)
        out.stats = {
            "success_count": out.success_count,
            "total_count": out.total_count,
            "total_attempts": sum(r.attempts for r in results),
        }
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _call_with_retries(
        self,
        client: httpx.AsyncClient,
        provider: Provider,
        prompt: str,
        images: list[Image],
    ) -> ProviderResult:
        last_error = ""
        max_attempts = self.max_retries + 1
        elapsed_total = 0

        for attempt in range(1, max_attempts + 1):
            start = time.monotonic()
            try:
                payload = provider.build_payload(prompt, images)
                response = await client.request(
                    payload.method,
                    payload.url,
                    headers=payload.headers,
                    json=payload.json_body,
                    timeout=provider.timeout_seconds,
                )
                elapsed_total += int((time.monotonic() - start) * 1000)

                if response.status_code >= 400:
                    last_error = (
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                    if attempt < max_attempts and is_retryable_http(response.status_code):
                        log.warning(
                            "%s attempt %d/%d: %s — retrying",
                            provider.name, attempt, max_attempts, last_error,
                        )
                        await linear_backoff_sleep(attempt, self.retry_backoff_ms)
                        continue
                    return ProviderResult(
                        provider=provider.name,
                        status=ResultStatus.FAILED,
                        error=last_error,
                        attempts=attempt,
                        elapsed_ms=elapsed_total,
                    )

                body = response.json()
                content = provider.parse_response(body)
                # Some gateways return HTTP 200 with empty content on transient
                # upstream errors instead of propagating the 5xx. Treat empty
                # content as retryable.
                if not content:
                    last_error = "Empty response content"
                    if attempt < max_attempts:
                        await linear_backoff_sleep(attempt, self.retry_backoff_ms)
                        continue

                return ProviderResult(
                    provider=provider.name,
                    status=ResultStatus.OK if content else ResultStatus.FAILED,
                    content=content,
                    error="" if content else last_error,
                    attempts=attempt,
                    elapsed_ms=elapsed_total,
                    raw=body,
                )

            except httpx.HTTPError as exc:
                elapsed_total += int((time.monotonic() - start) * 1000)
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_attempts and is_retryable_exception(exc):
                    log.warning(
                        "%s attempt %d/%d: %s — retrying",
                        provider.name, attempt, max_attempts, last_error,
                    )
                    await linear_backoff_sleep(attempt, self.retry_backoff_ms)
                    continue
                return ProviderResult(
                    provider=provider.name,
                    status=ResultStatus.FAILED,
                    error=last_error,
                    attempts=attempt,
                    elapsed_ms=elapsed_total,
                )

        return ProviderResult(
            provider=provider.name,
            status=ResultStatus.FAILED,
            error=last_error or "All retries exhausted",
            attempts=max_attempts,
            elapsed_ms=elapsed_total,
        )
