"""OpenAI-compatible chat-completions client (xAI Grok, Groq, OpenAI, vLLM, ...).

Implemented on ``requests`` so retry/backoff behaviour is explicit and testable
without any vendor SDK.
"""
from __future__ import annotations

import os
import random
import time
from typing import Callable, List, Optional

import requests

from genesis.llm.base import (
    LLMAuthError, LLMBadRequestError, LLMConnectionError, LLMError, LLMProvider,
    LLMRateLimitError, LLMRequest, LLMResponse, LLMResponseError, LLMServerError, Usage,
)
from genesis.llm.pricing import estimate_cost


class OpenAICompatProvider(LLMProvider):
    name = "openai-compatible"
    default_base_url = "https://api.openai.com/v1"
    default_api_key_env = "OPENAI_API_KEY"
    supports_json_schema = True

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        *,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        timeout: float = 180.0,
        pricing: Optional[dict] = None,
        session=None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ):
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.api_key_env = api_key_env or self.default_api_key_env
        self._api_key = api_key
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self.pricing = pricing
        self.session = session or requests.Session()
        self.sleep = sleep
        self.rng = rng or random.Random()

    # -- hooks subclasses can override ---------------------------------
    def accepts_reasoning_effort(self, model: str) -> bool:
        return False

    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        key = self._api_key or os.getenv(self.api_key_env)
        if not key:
            raise LLMAuthError(
                f"{self.api_key_env} is not set. Copy .env.example to .env and add your key, "
                f"or export {self.api_key_env} in your shell."
            )
        return key

    def _payload(self, request: LLMRequest) -> dict:
        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_completion_tokens": request.max_tokens,
        }
        if request.json_schema and self.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.json_schema["name"],
                    "schema": request.json_schema["schema"],
                    "strict": True,
                },
            }
        elif request.json_mode or request.json_schema:
            payload["response_format"] = {"type": "json_object"}
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.reasoning_effort and self.accepts_reasoning_effort(request.model):
            payload["reasoning_effort"] = request.reasoning_effort
        return payload

    @staticmethod
    def _retry_after(response) -> Optional[float]:
        value = (getattr(response, "headers", None) or {}).get("retry-after")
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        return seconds if seconds >= 0 else None

    def _error_for(self, response) -> LLMError:
        status = response.status_code
        try:
            body = response.json()
            detail = body.get("error", body) if isinstance(body, dict) else body
            if isinstance(detail, dict):
                detail = detail.get("message") or detail
        except Exception:
            detail = (getattr(response, "text", "") or "")[:300]
        message = f"{self.name} HTTP {status}: {detail}"
        retry_after = self._retry_after(response)
        if status == 429:
            return LLMRateLimitError(message, status, retry_after)
        if status in (401, 403):
            return LLMAuthError(message, status)
        if status == 408 or status >= 500:
            return LLMServerError(message, status, retry_after)
        return LLMBadRequestError(message, status)

    def _backoff(self, attempt: int, error: LLMError) -> float:
        if error.retry_after is not None:
            return min(error.retry_after, self.backoff_max * 2)
        delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        return delay + self.rng.random() * delay * 0.25

    def _single_call(self, request: LLMRequest) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        started = time.perf_counter()
        try:
            response = self.session.post(
                url, headers=headers, json=self._payload(request), timeout=self.timeout
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise LLMConnectionError(f"{self.name} connection failed: {exc}") from exc
        if response.status_code != 200:
            raise self._error_for(response)
        try:
            body = response.json()
            choice = body["choices"][0]
            content = (choice.get("message") or {}).get("content")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"{self.name} returned a malformed body: {exc}") from exc
        if not content or not str(content).strip():
            raise LLMResponseError(
                f"{self.name} returned empty content for {request.model} "
                f"(finish_reason={choice.get('finish_reason')}). Raise max_tokens for reasoning models."
            )
        raw_usage = body.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
            completion_tokens=int(raw_usage.get("completion_tokens") or 0),
            reasoning_tokens=int(
                (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            ),
            cached_tokens=int(
                (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            ),
            total_tokens=int(raw_usage.get("total_tokens") or 0),
        )
        model = body.get("model") or request.model
        cost = estimate_cost(model, usage, self.pricing)
        if cost is None:
            cost = estimate_cost(request.model, usage, self.pricing)
        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=choice.get("finish_reason"),
            cost_usd=cost,
            provider=self.name,
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        total_started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._single_call(request)
                response.attempts = attempt + 1
                response.latency_ms = (time.perf_counter() - total_started) * 1000
                return response
            except LLMError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                self.sleep(self._backoff(attempt, exc))
        raise LLMError("retry loop ended unexpectedly")  # pragma: no cover

    def list_models(self) -> List[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = self.session.get(f"{self.base_url}/models", headers=headers, timeout=30)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise LLMConnectionError(f"{self.name} connection failed: {exc}") from exc
        if response.status_code != 200:
            raise self._error_for(response)
        return [item["id"] for item in response.json().get("data", [])]


class XAIProvider(OpenAICompatProvider):
    """xAI Grok via the OpenAI-compatible API (https://api.x.ai/v1, XAI_API_KEY)."""

    name = "xai"
    default_base_url = "https://api.x.ai/v1"
    default_api_key_env = "XAI_API_KEY"

    # Only these families take ``reasoning_effort`` (grok-4.20 variants are fixed-mode
    # and reject it); sending it elsewhere is dropped client-side.
    _REASONING_EFFORT_PREFIXES = ("grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3")

    def accepts_reasoning_effort(self, model: str) -> bool:
        return (model or "").lower().startswith(self._REASONING_EFFORT_PREFIXES)


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    default_api_key_env = "GROQ_API_KEY"
    supports_json_schema = False  # json_object mode only

    def accepts_reasoning_effort(self, model: str) -> bool:
        return "gpt-oss" in (model or "").lower()


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_api_key_env = "OPENAI_API_KEY"

    def accepts_reasoning_effort(self, model: str) -> bool:
        return (model or "").lower().startswith(("o1", "o3", "o4", "gpt-5"))
