"""Build providers and resolve which provider/model/parameters each agent uses."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from genesis.llm.base import LLMProvider
from genesis.llm.mock import MockProvider
from genesis.llm.openai_compat import (
    GroqProvider, OpenAICompatProvider, OpenAIProvider, XAIProvider,
)

PROVIDERS = {
    "xai": XAIProvider,
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "openai-compatible": OpenAICompatProvider,
    "mock": MockProvider,
}

# Default model tiers. See docs/GROK.md for why these ids were chosen.
DEFAULT_LLM = {
    "provider": "xai",
    "default_tier": "main",
    "tiers": {
        "main": {"model": "grok-4.7", "reasoning_effort": "medium",
                 "temperature": 0.7, "max_tokens": 4096},
        "light": {"model": "grok-4.20-0309-non-reasoning",
                  "temperature": 0.7, "max_tokens": 1024},
    },
    "max_retries": 4,
    "max_repairs": 2,
    "max_cost_usd": None,
}
MOCK_TIERS = {
    "main": {"model": "mock-main", "temperature": 0.7, "max_tokens": 1024},
    "light": {"model": "mock-light", "temperature": 0.7, "max_tokens": 1024},
}


def load_env(path: Optional[Path] = None) -> None:
    """Load ``.env`` from the project root (never overrides real environment)."""
    env_path = path or Path(__file__).resolve().parents[2] / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    if env_path.exists():
        load_dotenv(env_path, override=False)


def merge_llm_config(*layers: Optional[dict]) -> dict:
    """Merge config layers (later wins); ``tiers`` merge per tier."""
    merged = {**DEFAULT_LLM, "tiers": {k: dict(v) for k, v in DEFAULT_LLM["tiers"].items()}}
    for layer in layers:
        for key, value in (layer or {}).items():
            if key == "tiers":
                for tier, params in value.items():
                    merged["tiers"].setdefault(tier, {}).update(params)
            else:
                merged[key] = value
    return merged


class ProviderPool:
    """Lazily creates one provider per (name, base_url) and resolves agent settings."""

    def __init__(self, llm_config: Optional[dict] = None, *, pricing: Optional[dict] = None,
                 providers: Optional[Dict[str, LLMProvider]] = None, mock_seed: int = 0,
                 mock_delay: float = 0.0):
        self.config = merge_llm_config(llm_config)
        self.pricing = pricing
        self._providers: Dict[str, LLMProvider] = dict(providers or {})
        self.mock_seed = mock_seed
        self.mock_delay = mock_delay

    def get(self, name: Optional[str] = None, **overrides) -> LLMProvider:
        name = name or self.config["provider"]
        key = f"{name}|{overrides.get('base_url') or ''}"
        if key not in self._providers:
            if name not in PROVIDERS:
                raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}")
            if name == "mock":
                self._providers[key] = MockProvider(seed=self.mock_seed, delay=self.mock_delay)
            else:
                options = {"max_retries": self.config.get("max_retries", 4), "pricing": self.pricing}
                options.update({k: v for k, v in overrides.items() if v is not None})
                self._providers[key] = PROVIDERS[name](**options)
        return self._providers[key]

    def resolve(self, agent: dict) -> dict:
        """Return ``{provider, model, params}`` for an agent definition.

        Precedence: agent ``llm`` block / ``model`` key  >  agent ``tier``  >  default tier.
        """
        llm = agent.get("llm") or {}
        provider_name = llm.get("provider") or agent.get("provider") or self.config["provider"]
        tiers = self.config["tiers"]
        if provider_name == "mock":
            tiers = {**tiers, **MOCK_TIERS}
        tier_name = llm.get("tier") or agent.get("tier") or self.config["default_tier"]
        if tier_name not in tiers:
            raise ValueError(f"Agent {agent.get('id')} uses unknown model tier '{tier_name}'.")
        params = dict(tiers[tier_name])
        model = llm.get("model") or agent.get("model") or params.pop("model")
        params.pop("model", None)
        params.update({k: v for k, v in llm.items()
                       if k not in ("provider", "tier", "model", "base_url")})
        if agent.get("model_parameters"):  # legacy per-agent parameters
            params.update(agent["model_parameters"])
        return {"provider": provider_name, "model": model, "params": params,
                "base_url": llm.get("base_url"), "tier": tier_name}
