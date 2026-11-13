import copy
import json

from genesis.engine import ScenarioRunner
from genesis.llm import MockProvider, ProviderPool
from genesis.scenario_spec import validate_scenario


def tiny_spec(**overrides):
    spec = {
        "id": "tiny", "title": "Tiny", "kind": "mixed", "summary": "s", "goal": "g", "briefing": "b",
        "turn_limit": 4,
        "resources": {"water": {"description": "w", "pool": 5, "agent_start": 2},
                      "labor": {"description": "l", "pool": 0, "agent_start": 2}},
        "variables": {"morale": {"initial": 0}},
        "health": {"initial": 100, "recovery_per_turn": 0},
        "upkeep": [{"resource": "water", "amount": 1, "source": "self_then_pool", "shortfall_health": -60}],
        "projects": {"well": {"description": "dig", "needs": {"labor": 3},
                              "on_complete": [{"target": "pool.water", "op": "add", "value": 4}]}},
        "votes": [{"id": "v1", "turn": 1, "window": 2, "question": "q", "default_option": "no",
                   "options": {"yes": {"description": "dy", "effects": [{"target": "var.morale", "op": "add", "value": 2}]},
                               "no": {"description": "dn", "effects": []}}}],
        "agents": [
            {"id": "a", "name": "Ann", "role": "r", "persona": "p", "goals": ["g"], "secret": "shh"},
            {"id": "b", "name": "Bob", "role": "r", "persona": "p", "goals": ["g"]},
        ],
        "rules": ["r"],
        "constraints": {"actions": ["pass", "say", "whisper", "give", "contribute", "draw", "propose",
                                    "accept", "reject", "vote", "take"],
                        "max_amount": 2, "draw_limit": {"water": 1, "labor": 0},
                        "take_success_rate": 1.0},
        "success": {"description": "d", "all": [
            {"id": "done", "description": "well dug", "metric": "project.well.done", "op": "==", "value": 1}]},
    }
    for key, value in overrides.items():
        if value is None:
            spec.pop(key, None)
        else:
            spec[key] = value
    if spec["turn_limit"] < 2:
        spec["votes"] = []
        spec["constraints"]["actions"] = [a for a in spec["constraints"]["actions"] if a != "vote"]
    return validate_scenario(spec)


def action(name, **fields):
    base = {"reasoning": "because", "action": name}
    base.update(fields)
    return base


def scripted_runner(spec, per_agent, tmp, seed=1, **kwargs):
    """per_agent: {agent_id: [reply per turn]} - replies are dicts or raw strings."""
    queues = {aid: list(replies) for aid, replies in per_agent.items()}

    def policy(request):
        aid = request.metadata["agent_id"]
        queue = queues.get(aid) or []
        return queue.pop(0) if queue else action("pass")

    pool = ProviderPool({"provider": "mock"}, providers={"mock|": MockProvider(policy=policy)})
    return ScenarioRunner(spec, pool, seed=seed, runs_dir=tmp, concurrency=1, **kwargs)
