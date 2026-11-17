#!/usr/bin/env python3
"""Data / config validation stage.

Checks society config, scenario packs, and environment contract files before
any experiment is allowed to run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis.scenario_spec import ScenarioError, list_scenarios  # noqa: E402
from genesis.scenarios import load_scenarios  # noqa: E402


REQUIRED_CONFIG_KEYS = {
    "days",
    "seed",
    "agents",
    "llm",
    "starting_food",
    "starting_community_food",
    "daily_food_supply",
    "daily_food_consumption",
}
REQUIRED_LLM_KEYS = {"provider", "tiers"}
REQUIRED_TIER_KEYS = {"model"}
AGENT_REQUIRED = {"id", "name", "role"}


def validate_society_config(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc})"]

    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        problems.append(f"{path}: missing keys {sorted(missing)}")

    agents = config.get("agents")
    if not isinstance(agents, list) or not agents:
        problems.append(f"{path}: agents must be a non-empty list")
    else:
        ids = []
        for i, agent in enumerate(agents):
            if not isinstance(agent, dict):
                problems.append(f"{path}: agents[{i}] must be an object")
                continue
            miss = AGENT_REQUIRED - set(agent)
            if miss:
                problems.append(f"{path}: agents[{i}] missing {sorted(miss)}")
            if "id" in agent:
                ids.append(agent["id"])
        if len(ids) != len(set(ids)):
            problems.append(f"{path}: duplicate agent ids")

    llm = config.get("llm")
    if not isinstance(llm, dict):
        problems.append(f"{path}: llm must be an object")
    else:
        miss = REQUIRED_LLM_KEYS - set(llm)
        if miss:
            problems.append(f"{path}: llm missing {sorted(miss)}")
        tiers = llm.get("tiers", {})
        if not isinstance(tiers, dict) or not tiers:
            problems.append(f"{path}: llm.tiers must be a non-empty object")
        else:
            for name, tier in tiers.items():
                if not isinstance(tier, dict):
                    problems.append(f"{path}: llm.tiers.{name} must be an object")
                    continue
                tmiss = REQUIRED_TIER_KEYS - set(tier)
                if tmiss:
                    problems.append(f"{path}: llm.tiers.{name} missing {sorted(tmiss)}")
    return problems


def validate_society_schedule(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        load_scenarios(path)
    except Exception as exc:  # noqa: BLE001 - surface any loader failure
        problems.append(f"{path}: {exc}")
    return problems


def validate_env_example(path: Path) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f"{path}: missing"]
    text = path.read_text(encoding="utf-8")
    if "XAI_API_KEY" not in text:
        problems.append(f"{path}: must document XAI_API_KEY")
    return problems


def validate_scenarios() -> list[str]:
    problems: list[str] = []
    try:
        scenarios = list_scenarios()
    except ScenarioError as exc:
        return [str(exc)]
    if not scenarios:
        problems.append("no scenarios found under scenarios/")
    for sid, spec in scenarios.items():
        if spec.get("turn_limit", 0) < 1:
            problems.append(f"scenario {sid}: turn_limit must be >= 1")
        if not spec.get("agents"):
            problems.append(f"scenario {sid}: needs at least one agent")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Genesis configs and scenario data.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    problems: list[str] = []
    problems.extend(validate_society_config(args.root / "config.json"))
    problems.extend(validate_society_schedule(args.root / "scenarios.json"))
    problems.extend(validate_env_example(args.root / ".env.example"))
    problems.extend(validate_scenarios())

    if problems:
        print("CONFIG VALIDATION FAILED")
        for item in problems:
            print(f"  - {item}")
        return 1

    scenarios = list_scenarios()
    print("CONFIG VALIDATION PASSED")
    print(f"  society config: ok")
    print(f"  society schedule: ok")
    print(f"  scenarios: {len(scenarios)} ({', '.join(sorted(scenarios))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
