import copy
import json
import tempfile
import unittest
from pathlib import Path

from genesis.scenario_spec import (
    ScenarioError, get_scenario, list_scenarios, load_scenario_file, validate_scenario,
)


def minimal():
    return {
        "id": "mini", "title": "Mini", "kind": "cooperation", "summary": "s", "goal": "g",
        "briefing": "b", "turn_limit": 3,
        "resources": {"water": {"description": "w", "pool": 4, "agent_start": 1}},
        "variables": {"morale": {"initial": 1}},
        "agents": [
            {"id": "a", "name": "A", "role": "r", "persona": "p", "goals": ["g"]},
            {"id": "b", "name": "B", "role": "r", "persona": "p", "goals": ["g"]},
        ],
        "rules": ["be nice"],
        "constraints": {"actions": ["pass", "say", "give"]},
        "success": {"description": "d", "all": [
            {"id": "c1", "description": "enough", "metric": "pool.water", "op": ">=", "value": 2}]},
    }


class ScenarioLoaderTests(unittest.TestCase):
    def problems(self, mutate):
        data = minimal()
        mutate(data)
        with self.assertRaises(ScenarioError) as ctx:
            validate_scenario(data)
        return "\n".join(ctx.exception.problems)

    def test_minimal_scenario_is_valid_and_defaults_are_filled(self):
        spec = validate_scenario(minimal())
        self.assertEqual(spec["constraints"]["max_amount"], 3)
        self.assertEqual(spec["on_timeout"], "failure")
        self.assertEqual(spec["agents"][0]["tier"], "main")
        self.assertEqual(spec["health"]["initial"], 100)

    def test_bundled_scenarios_validate_and_cover_each_kind(self):
        scenarios = list_scenarios()
        self.assertGreaterEqual(len(scenarios), 4)
        kinds = {s["kind"] for s in scenarios.values()}
        self.assertTrue({"cooperation", "competition", "decision-making"} <= kinds)
        for spec in scenarios.values():
            self.assertGreaterEqual(spec["turn_limit"], 1)
            self.assertTrue(spec.get("success") or spec.get("failure") or spec.get("scoring"))
            self.assertTrue(spec["rules"] and spec["goal"])
            for agent in spec["agents"]:
                self.assertTrue(agent["goals"] and agent["persona"])

    def test_rejects_unknown_resource_references_in_conditions_and_effects(self):
        text = self.problems(lambda d: d["success"]["all"][0].update(metric="pool.gold"))
        self.assertIn("unknown metric 'pool.gold'", text)
        text = self.problems(lambda d: d.update(events=[{
            "id": "e", "turn": 1, "title": "t", "description": "d",
            "effects": [{"target": "pool.gold", "op": "add", "value": 1}]}]))
        self.assertIn("unknown target 'pool.gold'", text)
        text = self.problems(lambda d: d["agents"][0].update(start={"gold": 1}))
        self.assertIn("unknown resource 'gold'", text)

    def test_rejects_bad_operator_duplicate_agent_and_turn_limit(self):
        self.assertIn("op:", self.problems(lambda d: d["success"]["all"][0].update(op="=~")))
        self.assertIn("duplicate agent id 'a'", self.problems(lambda d: d["agents"][1].update(id="a")))
        self.assertIn("turn_limit", self.problems(lambda d: d.update(turn_limit=0)))
        self.assertIn("turn_limit", self.problems(lambda d: d.update(turn_limit="ten")))

    def test_rejects_structural_mistakes(self):
        self.assertIn("kind:", self.problems(lambda d: d.update(kind="chaos")))
        self.assertIn("must include 'pass'", self.problems(lambda d: d["constraints"].update(actions=["say"])))
        self.assertIn("unknown action 'dance'", self.problems(lambda d: d["constraints"].update(actions=["pass", "dance"])))
        self.assertIn("'vote' needs", self.problems(lambda d: d["constraints"].update(actions=["pass", "vote"])))
        self.assertIn("at least 2 agents", self.problems(lambda d: d.update(agents=d["agents"][:1])))
        self.assertIn("need at least one of success", self.problems(lambda d: d.pop("success")))
        self.assertIn("events[0].turn", self.problems(lambda d: d.update(events=[{
            "id": "e", "turn": 9, "title": "t", "description": "d", "effects": []}])))

    def test_all_problems_are_reported_together(self):
        def many(d):
            d["turn_limit"] = 0
            d["kind"] = "x"
            d["success"]["all"][0]["metric"] = "nope"
        text = self.problems(many)
        self.assertGreaterEqual(text.count("\n") + 1, 3)

    def test_vote_default_must_exist_and_window_must_fit(self):
        vote = {"id": "v", "turn": 2, "window": 3, "question": "q", "default_option": "zzz",
                "options": {"x": {"description": "dx", "effects": []},
                            "y": {"description": "dy", "effects": []}}}
        text = self.problems(lambda d: d.update(votes=[vote]))
        self.assertIn("default_option", text)
        self.assertIn("window must close by turn_limit", text)

    def test_file_loading_selection_and_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.json"
            path.write_text(json.dumps(minimal()))
            self.assertEqual(load_scenario_file(path)["id"], "mini")
            self.assertEqual(get_scenario(str(path))["id"], "mini")
            self.assertEqual(get_scenario("mini", tmp)["id"], "mini")
            (Path(tmp) / "broken.json").write_text("{nope")
            with self.assertRaises(ScenarioError):
                list_scenarios(tmp)
        with self.assertRaises(KeyError):
            get_scenario("does-not-exist")
        self.assertEqual(get_scenario("ram-setu")["kind"], "cooperation")


if __name__ == "__main__":
    unittest.main()
