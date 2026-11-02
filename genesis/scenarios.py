import json
from pathlib import Path


EFFECT_KEYS = {
    "daily_food_supply_delta",
    "daily_medicine_supply_delta",
    "community_food_delta",
    "community_medicine_delta",
}


def load_scenarios(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Scenario configuration must be a JSON object.")
    if not isinstance(data.get("version"), str) or not data["version"].strip():
        raise ValueError("Scenario configuration must declare a non-empty version.")
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Scenario configuration must contain a 'scenarios' list.")
    seen_ids = set()
    seen_days = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Every scenario must be a JSON object.")
        scenario_id = scenario.get("id")
        day = scenario.get("day")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("Every scenario must have a non-empty string id.")
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario id: {scenario_id}")
        if isinstance(day, bool) or not isinstance(day, int) or day < 1:
            raise ValueError(f"Scenario {scenario_id} must have a positive integer day.")
        if day in seen_days:
            raise ValueError(f"Only one scheduled scenario is supported per day ({day}).")
        for field in ("title", "description"):
            if not isinstance(scenario.get(field), str) or not scenario[field].strip():
                raise ValueError(f"Scenario {scenario_id} must have a non-empty {field}.")
        effects = scenario.get("effects", {})
        if not isinstance(effects, dict) or set(effects) - EFFECT_KEYS:
            raise ValueError(f"Scenario {scenario_id} contains unsupported effects.")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in effects.values()
        ):
            raise ValueError(f"Scenario {scenario_id} effects must be integer values.")
        facts = scenario.get("facts", [])
        if not isinstance(facts, list) or any(
            not isinstance(fact, str) or not fact.strip() for fact in facts
        ):
            raise ValueError(f"Scenario {scenario_id} facts must be non-empty strings.")
        seen_ids.add(scenario_id)
        seen_days.add(day)
    return data


def scenario_for_day(scenarios, day):
    return next(
        (scenario for scenario in scenarios["scenarios"] if scenario["day"] == day),
        None,
    )


def apply_scenario(state, scenario):
    if scenario is None:
        state["active_scenario"] = None
        return {
            "daily_food_supply_delta": 0,
            "daily_medicine_supply_delta": 0,
        }

    effects = scenario.get("effects", {})
    state["community"]["food"] = max(
        0, state["community"]["food"] + effects.get("community_food_delta", 0)
    )
    state["community"]["medicine"] = max(
        0, state["community"]["medicine"] + effects.get("community_medicine_delta", 0)
    )
    state["active_scenario"] = {
        "id": scenario["id"],
        "title": scenario["title"],
        "description": scenario["description"],
        "facts": scenario.get("facts", []),
    }
    return {
        "daily_food_supply_delta": effects.get("daily_food_supply_delta", 0),
        "daily_medicine_supply_delta": effects.get("daily_medicine_supply_delta", 0),
    }
