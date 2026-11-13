"""Tightly scoped, config-driven scenarios: schema, validating loader and metric helpers.

A scenario is one JSON file in ``scenarios/`` describing a closed world: a goal, agent
roles and personas, rules and constraints, resources and variables, scheduled events,
votes, projects, success/failure conditions and a hard turn limit. Everything the
engine does is derived from this file, so adding a scenario needs no code change.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"
KINDS = ("cooperation", "competition", "decision-making", "mixed")
ACTION_NAMES = (
    "pass", "say", "whisper", "give", "contribute", "draw", "propose", "accept", "reject",
    "vote", "take",
)
OPS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}
UPKEEP_SOURCES = ("self", "pool", "self_then_pool")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEFAULTS_CONSTRAINTS = {
    "max_amount": 3,
    "draw_limit": {},
    "pool_visible": True,
    "agent_resources_visible": True,
    "message_max_chars": 300,
    "offer_expiry_turns": 2,
    "take_success_rate": 0.5,
    "take_penalty": 1,
}


class ScenarioError(ValueError):
    """Raised with every problem found in a scenario file, not just the first."""

    def __init__(self, source: str, problems: List[str]):
        self.source = source
        self.problems = problems
        super().__init__(f"Invalid scenario {source}:\n  - " + "\n  - ".join(problems))


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


class _Validator:
    def __init__(self, data: dict):
        self.data = data
        self.problems: List[str] = []
        self.resources: Dict[str, dict] = {}
        self.variables: Dict[str, dict] = {}
        self.agent_ids: List[str] = []
        self.project_ids: List[str] = []

    def err(self, message: str) -> None:
        self.problems.append(message)

    # -- reference helpers ---------------------------------------------------
    def check_metric(self, metric: Any, where: str) -> None:
        if not _text(metric):
            return self.err(f"{where}: metric must be a non-empty string")
        parts = metric.split(".")
        head = parts[0]
        ok = False
        if metric in ("turn", "agents.alive"):
            ok = True
        elif head == "pool" and len(parts) == 2:
            ok = parts[1] in self.resources
        elif head == "var" and len(parts) == 2:
            ok = parts[1] in self.variables
        elif head == "agent" and len(parts) == 3:
            ok = parts[1] in self.agent_ids and (parts[2] in self.resources or parts[2] == "health")
        elif head == "agents" and len(parts) == 3 and parts[1] in ("min", "max", "sum"):
            ok = parts[2] in self.resources or parts[2] == "health"
        elif head == "project" and len(parts) == 3:
            ok = parts[1] in self.project_ids and parts[2] in ("progress", "done")
        elif head == "score" and len(parts) == 2:
            ok = parts[1] in self.agent_ids or parts[1] in ("max", "min")
        if not ok:
            self.err(
                f"{where}: unknown metric '{metric}' (use pool.<resource>, var.<name>, "
                "agent.<id>.<resource|health>, agents.<min|max|sum>.<resource|health>, "
                "agents.alive, project.<id>.<progress|done>, score.<id|max|min>, turn)"
            )

    def check_effect(self, effect: Any, where: str) -> None:
        if not isinstance(effect, dict):
            return self.err(f"{where}: effect must be an object")
        if effect.get("op") not in ("add", "set"):
            self.err(f"{where}: op must be 'add' or 'set'")
        if not _is_int(effect.get("value")):
            self.err(f"{where}: value must be an integer")
        target = effect.get("target")
        if not _text(target):
            return self.err(f"{where}: target must be a string")
        parts = target.split(".")
        ok = False
        if parts[0] == "pool" and len(parts) == 2:
            ok = parts[1] in self.resources
        elif parts[0] == "var" and len(parts) == 2:
            ok = parts[1] in self.variables
        elif parts[0] == "agent" and len(parts) == 3:
            ok = parts[1] in self.agent_ids and (parts[2] in self.resources or parts[2] == "health")
        elif parts[0] == "agents" and len(parts) == 3 and parts[1] in ("all", "alive"):
            ok = parts[2] in self.resources or parts[2] == "health"
        if not ok:
            self.err(f"{where}: unknown target '{target}' (pool.<resource>, var.<name>, "
                     "agent.<id>.<res|health>, agents.<all|alive>.<res|health>)")

    def check_effects(self, effects: Any, where: str) -> None:
        if effects is None:
            return
        if not isinstance(effects, list):
            return self.err(f"{where}: effects must be a list")
        for i, effect in enumerate(effects):
            self.check_effect(effect, f"{where}[{i}]")

    def check_conditions(self, group: Any, where: str) -> None:
        if not isinstance(group, dict):
            return self.err(f"{where}: must be an object with 'description' and 'all' and/or 'any'")
        if not _text(group.get("description")):
            self.err(f"{where}.description: required")
        listed = False
        seen = set()
        for mode in ("all", "any"):
            conditions = group.get(mode)
            if conditions is None:
                continue
            listed = True
            if not isinstance(conditions, list) or not conditions:
                self.err(f"{where}.{mode}: must be a non-empty list")
                continue
            for i, cond in enumerate(conditions):
                at = f"{where}.{mode}[{i}]"
                if not isinstance(cond, dict):
                    self.err(f"{at}: must be an object")
                    continue
                cid = cond.get("id")
                if not _text(cid):
                    self.err(f"{at}.id: required")
                elif cid in seen:
                    self.err(f"{at}.id: duplicate condition id '{cid}'")
                seen.add(cid)
                if not _text(cond.get("description")):
                    self.err(f"{at}.description: required")
                if cond.get("op") not in OPS:
                    self.err(f"{at}.op: must be one of {', '.join(OPS)}")
                if isinstance(cond.get("value"), bool) or not isinstance(cond.get("value"), (int, float)):
                    self.err(f"{at}.value: must be a number")
                if cond.get("when", "any_turn") not in ("any_turn", "at_limit"):
                    self.err(f"{at}.when: must be 'any_turn' or 'at_limit'")
                self.check_metric(cond.get("metric"), f"{at}.metric")
        if not listed:
            self.err(f"{where}: needs a non-empty 'all' or 'any' condition list")

    # -- sections ------------------------------------------------------------
    def run(self) -> None:
        d = self.data
        if not isinstance(d, dict):
            return self.err("top level must be a JSON object")
        if not _text(d.get("id")) or not ID_RE.match(str(d.get("id"))):
            self.err("id: required, letters/digits/-/_ only")
        for key in ("title", "summary", "goal", "briefing"):
            if not _text(d.get(key)):
                self.err(f"{key}: required non-empty string")
        if d.get("kind") not in KINDS:
            self.err(f"kind: must be one of {', '.join(KINDS)}")
        if not _is_int(d.get("turn_limit")) or d.get("turn_limit", 0) < 1:
            self.err("turn_limit: required integer >= 1")
        limit = d.get("turn_limit") if _is_int(d.get("turn_limit")) else 10**6

        resources = d.get("resources")
        if not isinstance(resources, dict) or not resources:
            self.err("resources: required non-empty object")
        else:
            for name, spec in resources.items():
                if not ID_RE.match(name):
                    self.err(f"resources.{name}: invalid name")
                if not isinstance(spec, dict):
                    self.err(f"resources.{name}: must be an object")
                    continue
                if not _text(spec.get("description")):
                    self.err(f"resources.{name}.description: required")
                for field in ("pool", "agent_start"):
                    if not _is_int(spec.get(field, 0)) or spec.get(field, 0) < 0:
                        self.err(f"resources.{name}.{field}: must be an integer >= 0")
                self.resources[name] = spec
        variables = d.get("variables", {})
        if not isinstance(variables, dict):
            self.err("variables: must be an object")
        else:
            for name, spec in variables.items():
                if not isinstance(spec, dict) or not _is_int(spec.get("initial")):
                    self.err(f"variables.{name}: needs an integer 'initial'")
                else:
                    self.variables[name] = spec

        agents = d.get("agents")
        if not isinstance(agents, list) or len(agents) < 2:
            self.err("agents: need a list of at least 2 agents")
        else:
            seen = set()
            for i, agent in enumerate(agents):
                at = f"agents[{i}]"
                if not isinstance(agent, dict):
                    self.err(f"{at}: must be an object")
                    continue
                aid = agent.get("id")
                if not _text(aid) or not ID_RE.match(str(aid)):
                    self.err(f"{at}.id: required (letters/digits/-/_)")
                elif aid in seen:
                    self.err(f"{at}.id: duplicate agent id '{aid}'")
                else:
                    seen.add(aid)
                    self.agent_ids.append(aid)
                for key in ("name", "role", "persona"):
                    if not _text(agent.get(key)):
                        self.err(f"{at}.{key}: required")
                goals = agent.get("goals")
                if not isinstance(goals, list) or not goals or not all(_text(g) for g in goals):
                    self.err(f"{at}.goals: need a non-empty list of strings")
                if "secret" in agent and not _text(agent["secret"]):
                    self.err(f"{at}.secret: must be a non-empty string when present")
                if agent.get("tier") not in (None, "main", "light") and not isinstance(agent.get("tier"), str):
                    self.err(f"{at}.tier: must be a string")
                if agent.get("mock_disposition", "balanced") not in ("cooperative", "selfish", "balanced"):
                    self.err(f"{at}.mock_disposition: cooperative|selfish|balanced")
                for res, amount in (agent.get("start") or {}).items():
                    if res not in self.resources:
                        self.err(f"{at}.start: unknown resource '{res}'")
                    elif not _is_int(amount) or amount < 0:
                        self.err(f"{at}.start.{res}: must be an integer >= 0")

        health = d.get("health", {})
        if not isinstance(health, dict) or not _is_int(health.get("initial", 100)) \
                or health.get("initial", 100) < 1:
            self.err("health.initial: must be an integer >= 1")

        upkeep = d.get("upkeep", [])
        if not isinstance(upkeep, list):
            self.err("upkeep: must be a list")
        else:
            for i, item in enumerate(upkeep):
                at = f"upkeep[{i}]"
                if not isinstance(item, dict) or item.get("resource") not in self.resources:
                    self.err(f"{at}.resource: must be a declared resource")
                    continue
                if not _is_int(item.get("amount")) or item["amount"] < 1:
                    self.err(f"{at}.amount: integer >= 1")
                if item.get("source", "self_then_pool") not in UPKEEP_SOURCES:
                    self.err(f"{at}.source: one of {', '.join(UPKEEP_SOURCES)}")
                if not _is_int(item.get("shortfall_health", 0)) or item.get("shortfall_health", 0) > 0:
                    self.err(f"{at}.shortfall_health: integer <= 0")

        projects = d.get("projects", {})
        if not isinstance(projects, dict):
            self.err("projects: must be an object")
            projects = {}
        for pid, spec in projects.items():
            if not ID_RE.match(pid) or not isinstance(spec, dict):
                self.err(f"projects.{pid}: invalid")
                continue
            self.project_ids.append(pid)
        for pid, spec in projects.items():
            if not isinstance(spec, dict):
                continue
            at = f"projects.{pid}"
            if not _text(spec.get("description")):
                self.err(f"{at}.description: required")
            needs = spec.get("needs")
            if not isinstance(needs, dict) or not needs:
                self.err(f"{at}.needs: non-empty object of resource -> integer")
            else:
                for res, amount in needs.items():
                    if res not in self.resources:
                        self.err(f"{at}.needs: unknown resource '{res}'")
                    elif not _is_int(amount) or amount < 1:
                        self.err(f"{at}.needs.{res}: integer >= 1")
            self.check_effects(spec.get("on_complete"), f"{at}.on_complete")

        events = d.get("events", [])
        if not isinstance(events, list):
            self.err("events: must be a list")
            events = []
        seen = set()
        for i, event in enumerate(events):
            at = f"events[{i}]"
            if not isinstance(event, dict):
                self.err(f"{at}: must be an object")
                continue
            if not _text(event.get("id")) or event["id"] in seen:
                self.err(f"{at}.id: required and unique")
            seen.add(event.get("id"))
            if not _is_int(event.get("turn")) or not 1 <= event["turn"] <= limit:
                self.err(f"{at}.turn: integer within 1..turn_limit")
            for key in ("title", "description"):
                if not _text(event.get(key)):
                    self.err(f"{at}.{key}: required")
            if "repeat_every" in event and (not _is_int(event["repeat_every"]) or event["repeat_every"] < 1):
                self.err(f"{at}.repeat_every: integer >= 1")
            self.check_effects(event.get("effects"), f"{at}.effects")

        votes = d.get("votes", [])
        if not isinstance(votes, list):
            self.err("votes: must be a list")
            votes = []
        seen = set()
        for i, vote in enumerate(votes):
            at = f"votes[{i}]"
            if not isinstance(vote, dict):
                self.err(f"{at}: must be an object")
                continue
            if not _text(vote.get("id")) or vote["id"] in seen:
                self.err(f"{at}.id: required and unique")
            seen.add(vote.get("id"))
            if not _is_int(vote.get("turn")) or not 1 <= vote["turn"] <= limit:
                self.err(f"{at}.turn: integer within 1..turn_limit")
            if not _is_int(vote.get("window", 1)) or vote.get("window", 1) < 1:
                self.err(f"{at}.window: integer >= 1")
            elif _is_int(vote.get("turn")) and vote["turn"] + vote.get("window", 1) - 1 > limit:
                self.err(f"{at}: voting window must close by turn_limit")
            if not _text(vote.get("question")):
                self.err(f"{at}.question: required")
            options = vote.get("options")
            if not isinstance(options, dict) or len(options) < 2:
                self.err(f"{at}.options: need at least 2 options")
                continue
            for name, option in options.items():
                if not isinstance(option, dict) or not _text(option.get("description")):
                    self.err(f"{at}.options.{name}.description: required")
                    continue
                self.check_effects(option.get("effects"), f"{at}.options.{name}.effects")
            if vote.get("default_option") not in options:
                self.err(f"{at}.default_option: must be one of the options")

        rules = d.get("rules")
        if not isinstance(rules, list) or not rules or not all(_text(r) for r in rules):
            self.err("rules: required non-empty list of strings")

        constraints = d.get("constraints", {})
        if not isinstance(constraints, dict):
            self.err("constraints: must be an object")
            constraints = {}
        actions = constraints.get("actions")
        if not isinstance(actions, list) or not actions:
            self.err("constraints.actions: required non-empty list")
        else:
            for name in actions:
                if name not in ACTION_NAMES:
                    self.err(f"constraints.actions: unknown action '{name}' "
                             f"(valid: {', '.join(ACTION_NAMES)})")
            if "pass" not in actions:
                self.err("constraints.actions: must include 'pass'")
            if "vote" in actions and not votes:
                self.err("constraints.actions: 'vote' needs at least one entry in votes")
        if not _is_int(constraints.get("max_amount", 3)) or constraints.get("max_amount", 3) < 1:
            self.err("constraints.max_amount: integer >= 1")
        for res, amount in (constraints.get("draw_limit") or {}).items():
            if res not in self.resources or not _is_int(amount) or amount < 0:
                self.err(f"constraints.draw_limit.{res}: declared resource, integer >= 0")
        rate = constraints.get("take_success_rate", 0.5)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            self.err("constraints.take_success_rate: number in 0..1")

        for key in ("success", "failure"):
            if key in d:
                self.check_conditions(d[key], key)
        if "success" not in d and "failure" not in d and "scoring" not in d:
            self.err("need at least one of success, failure or scoring conditions")
        if d.get("on_timeout", "failure") not in ("failure", "success", "score"):
            self.err("on_timeout: failure|success|score")
        scoring = d.get("scoring")
        if scoring is not None:
            weights = scoring.get("weights") if isinstance(scoring, dict) else None
            if not isinstance(weights, dict) or not weights:
                self.err("scoring.weights: non-empty object of resource -> number")
            else:
                for res, w in weights.items():
                    if res not in self.resources or isinstance(w, bool) or not isinstance(w, (int, float)):
                        self.err(f"scoring.weights.{res}: declared resource, numeric weight")
        elif d.get("on_timeout") == "score":
            self.err("on_timeout 'score' requires a scoring section")


def validate_scenario(data: Any, source: str = "<memory>") -> dict:
    """Validate and normalise a scenario dict; raises :class:`ScenarioError`."""
    validator = _Validator(data)
    validator.run()
    if validator.problems:
        raise ScenarioError(source, validator.problems)
    spec = json.loads(json.dumps(data))  # deep copy
    spec["constraints"] = {**DEFAULTS_CONSTRAINTS, **spec.get("constraints", {})}
    spec.setdefault("variables", {})
    spec.setdefault("projects", {})
    spec.setdefault("events", [])
    spec.setdefault("votes", [])
    spec.setdefault("upkeep", [])
    spec.setdefault("on_timeout", "score" if spec.get("scoring") else "failure")
    spec["health"] = {"initial": 100, "recovery_per_turn": 0, **spec.get("health", {})}
    for agent in spec["agents"]:
        agent.setdefault("tier", "main")
        agent.setdefault("start", {})
        agent.setdefault("mock_disposition", "balanced")
    return spec


def load_scenario_file(path) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ScenarioError(str(path), [f"not valid JSON: {exc}"]) from exc
    return validate_scenario(data, str(path))


def list_scenarios(directory=None) -> Dict[str, dict]:
    """Load and validate every ``*.json`` in the scenario directory, keyed by id."""
    found: Dict[str, dict] = {}
    for path in sorted(Path(directory or SCENARIO_DIR).glob("*.json")):
        spec = load_scenario_file(path)
        if spec["id"] in found:
            raise ScenarioError(str(path), [f"duplicate scenario id '{spec['id']}'"])
        spec["_path"] = str(path)
        found[spec["id"]] = spec
    return found


def get_scenario(name_or_path, directory=None) -> dict:
    """Resolve a scenario by id, or by path to a JSON file."""
    candidate = Path(str(name_or_path))
    if candidate.suffix == ".json" and candidate.exists():
        return load_scenario_file(candidate)
    scenarios = list_scenarios(directory)
    if name_or_path not in scenarios:
        raise KeyError(
            f"Unknown scenario '{name_or_path}'. Available: {', '.join(scenarios) or '(none)'}"
        )
    return scenarios[name_or_path]
