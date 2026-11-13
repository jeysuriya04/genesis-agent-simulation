"""Ask a provider for structured JSON, validate it, and repair invalid replies."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from genesis.llm.base import LLMProvider, LLMRequest, LLMResponse, Usage


def extract_json_object(raw: str) -> Any:
    """Parse a JSON object from model text, tolerating code fences and prose."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


@dataclass
class StructuredResult:
    value: Any = None
    raw: str = ""
    errors: List[str] = field(default_factory=list)
    repairs: int = 0
    calls: int = 0
    retries: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: Optional[float] = None
    latency_ms: float = 0.0
    model: str = ""
    responses: List[LLMResponse] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.errors


def request_structured(
    provider: LLMProvider,
    request: LLMRequest,
    parse: Callable[[str], Tuple[Any, List[str]]],
    max_repairs: int = 2,
) -> StructuredResult:
    """Call ``provider`` and run ``parse(raw) -> (value, errors)``.

    When ``errors`` is non-empty the model is shown its own reply plus the errors
    and asked again, up to ``max_repairs`` extra calls. Transport-level retries
    and rate-limit backoff happen inside the provider; an :class:`LLMError` that
    survives them propagates to the caller.
    """
    result = StructuredResult(model=request.model)
    messages = list(request.messages)
    costs: List[float] = []
    for attempt in range(max_repairs + 1):
        call = LLMRequest(**{**request.__dict__, "messages": messages})
        response = provider.complete(call)
        result.responses.append(response)
        result.calls += 1
        result.retries += max(0, response.attempts - 1)
        result.usage.add(response.usage)
        result.latency_ms += response.latency_ms
        result.model = response.model or result.model
        if response.cost_usd is not None:
            costs.append(response.cost_usd)
        result.raw = response.content
        value, errors = parse(response.content)
        result.value, result.errors = value, errors
        if not errors:
            break
        if attempt < max_repairs:
            result.repairs += 1
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": "Your last reply was rejected: " + "; ".join(errors)
                    + ". Reply again with ONLY one valid JSON object that fixes this.",
                },
            ]
    result.cost_usd = sum(costs) if costs else None
    return result
