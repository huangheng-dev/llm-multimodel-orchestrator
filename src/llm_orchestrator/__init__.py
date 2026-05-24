"""llm-multimodel-orchestrator

Parallel multi-LLM orchestration with retry, consensus voting, and
provider-agnostic abstraction.
"""
from .consensus import ConsensusStrategy, FirstSuccess, MajorityVote
from .orchestrator import Orchestrator
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    Provider,
    RequestPayload,
)
from .types import (
    Image,
    OrchestrationResult,
    ProviderResult,
    ResultStatus,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Orchestrator",
    # Providers
    "Provider",
    "RequestPayload",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    # Consensus
    "ConsensusStrategy",
    "FirstSuccess",
    "MajorityVote",
    # Data models
    "Image",
    "OrchestrationResult",
    "ProviderResult",
    "ResultStatus",
]
