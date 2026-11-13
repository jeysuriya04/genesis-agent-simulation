"""Provider-neutral request/response types and error taxonomy for LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class LLMError(Exception):
    """Base class for backend failures. ``retryable`` drives the backoff loop."""

    retryable = False

    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class LLMRateLimitError(LLMError):
    retryable = True


class LLMServerError(LLMError):
    retryable = True


class LLMConnectionError(LLMError):
    retryable = True


class LLMAuthError(LLMError):
    """Missing/invalid API key or no access to the model. Never retried."""


class LLMBadRequestError(LLMError):
    """The provider rejected the request itself (bad params, unknown model)."""


class LLMResponseError(LLMError):
    """The provider answered 200 but with an unusable body (e.g. empty content)."""

    retryable = True


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    @property
    def output_tokens(self) -> int:
        """Billable output tokens.

        xAI reports reasoning tokens separately from ``completion_tokens`` and
        ``total_tokens`` covers both, so derive output from the total when we can.
        """
        if self.total_tokens:
            return max(self.completion_tokens, self.total_tokens - self.prompt_tokens)
        return self.completion_tokens + self.reasoning_tokens

    def add(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens
        self.total_tokens += other.total_tokens
        return self

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class LLMRequest:
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2048
    # {"name": str, "schema": dict}: ask for schema-constrained JSON when supported.
    json_schema: Optional[Dict[str, Any]] = None
    json_mode: bool = True
    reasoning_effort: Optional[str] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    # Free-form context for providers that need it (the mock policy reads it).
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    attempts: int = 1
    finish_reason: Optional[str] = None
    cost_usd: Optional[float] = None
    provider: str = ""


class LLMProvider(ABC):
    name = "provider"
    supports_json_schema = False

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one completion or raise an :class:`LLMError`."""

    def list_models(self) -> List[str]:
        raise NotImplementedError(f"{self.name} cannot list models")
