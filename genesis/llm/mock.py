"""Deterministic mock LLM so scenarios, tests and demos run with no API key."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Callable, Dict, List, Optional

from genesis.llm.base import (
    LLMProvider, LLMRateLimitError, LLMRequest, LLMResponse, Usage,
)

# Relative preference per action for each disposition (unknown actions weigh 1).
DISPOSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "cooperative": {"contribute": 6, "give": 3, "say": 2, "vote": 3, "propose": 2, "accept": 5,
                    "draw": 1, "take": 0.05, "pass": 0.5, "whisper": 1, "reject": 0.5},
    "selfish": {"contribute": 0.8, "give": 0.3, "say": 1, "vote": 2, "propose": 2, "accept": 1,
                "draw": 6, "take": 3, "pass": 1, "whisper": 2, "reject": 3},
    "balanced": {"contribute": 3, "give": 1.5, "say": 1.5, "vote": 3, "propose": 2, "accept": 3,
                 "draw": 3, "take": 0.4, "pass": 1, "whisper": 1, "reject": 1.5},
}
_FIELDS = ("target", "resource", "amount", "request_resource", "request_amount",
           "offer_id", "vote_id", "option", "project", "message")


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockProvider(LLMProvider):
    """Returns scripted replies, or picks a legal action from ``metadata['candidates']``.

    ``fail_first`` raises that many rate-limit errors first (retry/backoff tests).
    """

    name = "mock"
    supports_json_schema = True

    def __init__(self, seed: int = 0, script: Optional[List[object]] = None,
                 policy: Optional[Callable[[LLMRequest], object]] = None, fail_first: int = 0):
        self.seed = seed
        self.script = list(script or [])
        self.policy = policy
        self.fail_first = fail_first
        self.requests: List[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise LLMRateLimitError("mock rate limit", status=429)
        if self.script:
            reply = self.script.pop(0)
        elif self.policy:
            reply = self.policy(request)
        else:
            reply = self.default_policy(request)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        prompt = sum(_approx_tokens(m["content"]) for m in request.messages)
        out = _approx_tokens(content)
        return LLMResponse(
            content=content, model=request.model or "mock",
            usage=Usage(prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out),
            latency_ms=1.0, cost_usd=0.0, provider="mock",
        )

    def default_policy(self, request: LLMRequest) -> dict:
        meta = request.metadata or {}
        candidates = meta.get("candidates") or [{"action": "pass"}]
        digest = hashlib.sha256(
            f"{self.seed}:{meta.get('agent_id')}:{meta.get('turn')}".encode()
        ).hexdigest()
        rng = random.Random(int(digest[:12], 16))
        weights = DISPOSITION_WEIGHTS.get(meta.get("disposition") or "balanced",
                                          DISPOSITION_WEIGHTS["balanced"])
        pick = rng.choices(candidates, [weights.get(c["action"], 1.0) for c in candidates])[0]
        reply = {name: pick.get(name) for name in _FIELDS}
        reply["action"] = pick["action"]
        reply["reasoning"] = pick.get("why") or (
            f"[mock:{meta.get('disposition') or 'balanced'}] chose {pick['action']} "
            f"on turn {meta.get('turn')}."
        )
        if reply["message"] is None and pick["action"] in ("say", "whisper", "propose"):
            reply["message"] = "Let's keep this fair and move together."
        return reply


def legacy_mock_decision(agent, context):
    """Valid decision for the original five-resident society (no API key needed)."""
    others = [o["id"] for o in context["other_agents"]]
    day = context["day"]
    rng = random.Random(f"{agent['id']}:{day}")
    action = rng.choice(["observe", "share", "request_shared", "govern", "observe", "help"])
    decision = {"action": action, "message": f"{agent['name']} acts on day {day}.",
                "reason": "mock policy", "resource": "food", "amount": 1,
                "target": rng.choice(others) if others else None}
    if action == "govern":
        decision["message"] = "Let us record what each resident needs before distributing food."
    return {"raw": json.dumps(decision), "model_version": "mock", "prompt_hash": None,
            "prompt_json": {}, "model_parameters": {},
            "usage": Usage().to_dict(), "cost_usd": 0.0}
