"""Token pricing (USD per 1M tokens) and thread-safe usage/cost tracking.

Rates are from https://docs.x.ai/docs/models (checked 2026-11-14). Override or extend
them with the ``pricing`` object in ``config.json``; unknown models report ``None``
cost rather than a guess.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

from genesis.llm.base import Usage

# prefix -> (input, cached input, output) per 1M tokens, prompt < 200k tokens
DEFAULT_PRICING: Dict[str, Tuple[float, float, float]] = {
    "grok-4.7": (2.00, 0.50, 6.00),
    "grok-4.6": (2.00, 0.50, 6.00),
    "grok-4.5": (2.00, 0.30, 6.00),
    "grok-4.3": (1.25, 0.20, 2.50),
    "grok-4.20": (1.25, 0.20, 2.50),
    "grok-build-0.1": (1.00, 0.20, 2.00),
    "mock": (0.0, 0.0, 0.0),
}
LONG_CONTEXT_THRESHOLD = 200_000  # requests at/above this are billed at 2x for all tokens


def resolve_rates(model: str, overrides: Optional[dict] = None):
    table = dict(DEFAULT_PRICING)
    for key, value in (overrides or {}).items():
        table[key] = tuple(value)
    name = (model or "").lower()
    best = None
    for prefix in table:
        if name.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return table[best] if best else None


def estimate_cost(model: str, usage: Usage, overrides: Optional[dict] = None) -> Optional[float]:
    rates = resolve_rates(model, overrides)
    if rates is None:
        return None
    rate_in, rate_cached, rate_out = rates
    if usage.prompt_tokens >= LONG_CONTEXT_THRESHOLD:
        rate_in, rate_cached, rate_out = rate_in * 2, rate_cached * 2, rate_out * 2
    cached = min(usage.cached_tokens, usage.prompt_tokens)
    fresh = usage.prompt_tokens - cached
    return (fresh * rate_in + cached * rate_cached + usage.output_tokens * rate_out) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


class UsageTracker:
    """Aggregates tokens, cost, latency and retries per agent and per model."""

    def __init__(self, max_cost_usd: Optional[float] = None):
        self.max_cost_usd = max_cost_usd
        self._lock = threading.Lock()
        self.total = Usage()
        self.cost_usd = 0.0
        self.unpriced_calls = 0
        self.calls = 0
        self.retries = 0
        self.by_agent: Dict[str, dict] = {}
        self.by_model: Dict[str, dict] = {}

    def record(self, agent_id: str, model: str, usage: Usage, cost: Optional[float],
               attempts: int = 1, latency_ms: float = 0.0) -> None:
        with self._lock:
            self.calls += 1
            self.retries += max(0, attempts - 1)
            self.total.add(usage)
            if cost is None:
                self.unpriced_calls += 1
            else:
                self.cost_usd += cost
            for bucket, key in ((self.by_agent, agent_id), (self.by_model, model)):
                row = bucket.setdefault(
                    key, {"calls": 0, "cost_usd": 0.0, "latency_ms": 0.0, "usage": Usage()}
                )
                row["calls"] += 1
                row["cost_usd"] += cost or 0.0
                row["latency_ms"] += latency_ms
                row["usage"].add(usage)

    def over_budget(self) -> bool:
        return self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd

    def summary(self) -> dict:
        def pack(bucket):
            return {
                key: {
                    "calls": row["calls"],
                    "cost_usd": round(row["cost_usd"], 6),
                    "avg_latency_ms": round(row["latency_ms"] / row["calls"], 1) if row["calls"] else 0,
                    **row["usage"].to_dict(),
                }
                for key, row in bucket.items()
            }

        with self._lock:
            return {
                "calls": self.calls,
                "retries": self.retries,
                "cost_usd": round(self.cost_usd, 6),
                "unpriced_calls": self.unpriced_calls,
                "max_cost_usd": self.max_cost_usd,
                **self.total.to_dict(),
                "by_agent": pack(self.by_agent),
                "by_model": pack(self.by_model),
            }
