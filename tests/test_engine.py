import json
import tempfile
import unittest

from genesis.events import EventLog, iter_runs, read_events, summarize_run
from genesis.llm import LLMAuthError, MockProvider, ProviderPool
from genesis.llm.base import LLMRateLimitError
from genesis.engine import ScenarioRunner
from genesis.scenario_spec import get_scenario, list_scenarios
from tests.helpers import action, scripted_runner, tiny_spec


def types(events):
    return [e["type"] for e in events]


class EngineRuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_script(self, spec, per_agent, **kwargs):
        runner = scripted_runner(spec, per_agent, self.tmp.name, **kwargs)
        result = runner.run()
        return result, read_events(result["log_path"])

    def steps(self, events, agent=None):
        return [e for e in events if e["type"] == "agent_step" and (agent is None or e["agent_id"] == agent)]

    def test_cooperative_project_completes_early_with_success(self):
        spec = tiny_spec()
        script = {"a": [action("contribute", resource="labor", amount=2, project="well")],
                  "b": [action("contribute", resource="labor", amount=1, project="well")]}
        result, events = self.run_script(spec, script)
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["turns_played"], 1)
        end = events[-1]
        self.assertEqual(end["type"], "run_end")
        self.assertEqual(end["state"]["projects"]["well"]["done"], True)
        self.assertEqual(end["state"]["pool"]["water"], 5 + 4)  # completion bonus; upkeep came from own stock first
        completion = [s for s in self.steps(events) if s["effect"]["ok"] and "project" in s["effect"]["summary"]]
        self.assertTrue(completion)

    def test_project_contribution_is_capped_at_remaining_need(self):
        spec = tiny_spec()
        script = {"a": [action("contribute", resource="labor", amount=2, project="well")],
                  "b": [action("contribute", resource="labor", amount=2, project="well")]}
        result, events = self.run_script(spec, script, seed=3)
        state = events[-1]["state"]
        self.assertEqual(state["projects"]["well"]["progress"]["labor"], 3)
        spent = sum(2 - v["resources"]["labor"] for v in state["agents"].values())
        self.assertEqual(spent, 3)

    def test_failure_when_agents_starve_and_estate_returns_to_pool(self):
        spec = tiny_spec(resources={"water": {"description": "w", "pool": 0, "agent_start": 0},
                                    "labor": {"description": "l", "pool": 0, "agent_start": 1}},
                         turn_limit=5)
        spec["failure"] = {"description": "d", "any": [
            {"id": "dead", "description": "one left", "metric": "agents.alive", "op": "<=", "value": 0}]}
        result, events = self.run_script(spec, {})
        self.assertEqual(result["outcome"], "failure")
        self.assertEqual(result["turns_played"], 2)  # -60 health per short turn: dead after two
        self.assertEqual(events[-1]["state"]["pool"]["labor"], 2)  # estates moved to the commons

    def test_timeout_when_goal_unmet(self):
        result, events = self.run_script(tiny_spec(turn_limit=2), {})
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["turns_played"], 2)

    def test_vote_plurality_default_and_effects(self):
        spec = tiny_spec()
        script = {"a": [action("vote", vote_id="v1", option="yes")],
                  "b": [action("vote", vote_id="v1", option="no"), action("vote", vote_id="v1", option="yes")]}
        result, events = self.run_script(spec, script, seed=2)
        results = [e for e in events if e["type"] == "world_event" and e["event_id"] == "vote:v1"]
        self.assertEqual(len(results), 1)
        self.assertEqual(events[-1]["state"]["votes"]["v1"]["result"], "yes")  # b changed their ballot
        self.assertEqual(events[-1]["state"]["variables"]["morale"], 2)
        # a tie falls back to the default option
        script = {"a": [action("vote", vote_id="v1", option="yes")],
                  "b": [action("vote", vote_id="v1", option="no")]}
        _, events = self.run_script(spec, script, seed=2)
        self.assertEqual(events[-1]["state"]["votes"]["v1"]["result"], "no")

    def test_trade_requires_acceptance_next_turn_and_swaps_atomically(self):
        spec = tiny_spec()
        script = {"a": [action("propose", target="b", resource="water", amount=1,
                               request_resource="labor", request_amount=2)],
                  "b": [action("pass"), action("accept", offer_id="O1-1")]}
        result, events = self.run_script(spec, script, seed=4)
        accept = [s for s in self.steps(events, "b") if s["action"]["action"] == "accept"][0]
        self.assertTrue(accept["effect"]["ok"], accept["effect"])
        self.assertEqual(accept["effect"]["interactions"][0]["type"], "accept")
        self.assertEqual(accept["perception"]["offers_to_you"][0]["id"], "O1-1")

    def test_draw_limit_and_theft_privacy(self):
        spec = tiny_spec()
        script = {"a": [action("take", target="b", resource="water", amount=2)],
                  "b": [action("draw", resource="water", amount=1)]}
        result, events = self.run_script(spec, script, seed=5)
        take = self.steps(events, "a")[0]
        self.assertTrue(take["effect"]["ok"])
        self.assertEqual(take["effect"]["interactions"][0]["type"], "take")
        turn2_b = [s for s in self.steps(events, "b") if s["turn"] == 2]
        self.assertTrue(turn2_b)
        msgs = turn2_b[0]["perception"]["private_messages_to_you"]
        self.assertTrue(any("Someone took" in m["message"] for m in msgs))  # victim is told, not who

    def test_secrets_are_visible_only_to_their_owner(self):
        _, events = self.run_script(tiny_spec(turn_limit=1), {})
        steps = {s["agent_id"]: s for s in self.steps(events)}
        self.assertEqual(steps["a"]["perception"]["you"]["secret_briefing"], "shh")
        self.assertNotIn("secret_briefing", steps["b"]["perception"]["you"])
        self.assertNotIn("shh", json.dumps(steps["b"]["perception"]))

    def test_competition_scoring_decides_winner_at_turn_limit(self):
        spec = tiny_spec(kind="competition", turn_limit=2, scoring={"weights": {"water": 1, "labor": 3}},
                         success=None)
        script = {"a": [action("give", target="b", resource="labor", amount=2)]}
        result, events = self.run_script(spec, script)
        self.assertEqual(result["outcome"], "decided")
        self.assertEqual(result["winner"], "b")
        self.assertGreater(result["scores"]["b"], result["scores"]["a"])
        self.assertEqual(events[-1]["winner"], "b")

    def test_invalid_reply_is_repaired_then_falls_back_to_pass(self):
        spec = tiny_spec(turn_limit=1)
        script = {"a": ["not json", action("say", message="fixed after repair")],
                  "b": ["bad", "worse", "still bad"]}
        _, events = self.run_script(spec, script)
        a, b = self.steps(events, "a")[0], self.steps(events, "b")[0]
        self.assertEqual(a["action"]["action"], "say")
        self.assertEqual(a["llm"]["repairs"], 1)
        self.assertEqual(b["action"]["action"], "pass")
        self.assertIn("FALLBACK_PASS", b["validation"]["flags"])
        self.assertTrue(b["validation"]["errors"])

    def test_illegal_move_is_rejected_by_validation_not_executed(self):
        spec = tiny_spec(turn_limit=1)
        script = {"a": [action("give", target="b", resource="water", amount=9), action("pass")]}
        _, events = self.run_script(spec, script)
        a = self.steps(events, "a")[0]
        self.assertEqual(a["action"]["action"], "pass")
        self.assertEqual(a["llm"]["repairs"], 1)

    def test_transient_llm_failure_becomes_pass_but_auth_error_aborts(self):
        spec = tiny_spec(turn_limit=1)

        class Flaky(MockProvider):
            def complete(self, request):
                if request.metadata["agent_id"] == "a":
                    raise LLMRateLimitError("still rate limited", 429)
                return super().complete(request)

        pool = ProviderPool({"provider": "mock"}, providers={"mock|": Flaky()})
        result = ScenarioRunner(spec, pool, runs_dir=self.tmp.name, concurrency=1).run()
        events = read_events(result["log_path"])
        a = self.steps(events, "a")[0]
        self.assertIn("LLM_ERROR", a["validation"]["flags"])
        self.assertIn("RateLimit", a["llm"]["error"])

        class NoKey(MockProvider):
            def complete(self, request):
                raise LLMAuthError("XAI_API_KEY is not set")

        pool = ProviderPool({"provider": "mock"}, providers={"mock|": NoKey()})
        runner = ScenarioRunner(spec, pool, runs_dir=self.tmp.name, concurrency=1)
        with self.assertRaises(LLMAuthError):
            runner.run()
        self.assertEqual(read_events(runner.log.path)[-1]["outcome"], "error")

    def test_cost_cap_stops_the_run(self):
        spec = tiny_spec(turn_limit=4)
        pool = ProviderPool({"provider": "mock", "max_cost_usd": 0.0000001})

        class Costly(MockProvider):
            def complete(self, request):
                response = super().complete(request)
                response.cost_usd = 0.01
                return response

        pool = ProviderPool({"provider": "mock", "max_cost_usd": 0.015},
                            providers={"mock|": Costly()})
        result = ScenarioRunner(spec, pool, runs_dir=self.tmp.name, concurrency=1).run()
        self.assertEqual(result["outcome"], "aborted_budget")
        self.assertEqual(result["turns_played"], 1)


class MockScenarioRunTests(unittest.TestCase):
    def test_every_bundled_scenario_runs_end_to_end_and_is_deterministic(self):
        for sid in list_scenarios():
            logs = []
            for _ in range(2):
                with tempfile.TemporaryDirectory() as tmp:
                    pool = ProviderPool({"provider": "mock"}, mock_seed=3)
                    result = ScenarioRunner(get_scenario(sid), pool, seed=3, runs_dir=tmp, run_id="r").run()
                    events = read_events(result["log_path"])
                    logs.append([(e["type"], e.get("turn"), (e.get("action") or {}).get("action"),
                                  e.get("outcome")) for e in events])
                    self.assertIn(result["outcome"], {"success", "failure", "timeout", "decided"})
            self.assertEqual(logs[0], logs[1], sid)


class EventLogTests(unittest.TestCase):
    def test_event_sequence_ordering_and_replay_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = ProviderPool({"provider": "mock"})
            result = ScenarioRunner(get_scenario("crater-council"), pool, seed=0, runs_dir=tmp).run()
            events = read_events(result["log_path"])
            self.assertEqual(types(events)[0], "run_start")
            self.assertEqual(types(events)[-1], "run_end")
            self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
            self.assertEqual(len({e["run_id"] for e in events}), 1)
            self.assertEqual(events, read_events(result["log_path"]))  # replay is lossless
            self.assertEqual(len(read_events(result["log_path"], after=10)), len(events) - 10)
            # Per turn: turn_start, [world_event]*, agent_step*, turn_end - in that order.
            current = None
            for event in events[1:-1]:
                if event["type"] == "turn_start":
                    self.assertIsNone(current)
                    current = event["turn"]
                elif event["type"] == "turn_end":
                    self.assertEqual(current, event["turn"])
                    current = None
                else:
                    self.assertEqual(current, event["turn"])
            steps = [e for e in events if e["type"] == "agent_step"]
            self.assertEqual(len(steps), 5 * result["turns_played"])
            for step in steps:
                for key in ("perception", "reasoning", "action", "effect", "llm", "validation"):
                    self.assertIn(key, step)
                self.assertIn("changes", step["effect"])
            summary = summarize_run(next(iter_runs(tmp)))
            self.assertEqual(summary["status"], result["outcome"])
            self.assertEqual(summary["turns_done"], result["turns_played"])

    def test_partial_trailing_line_is_ignored_for_live_readers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog("live", tmp)
            log.emit("run_start", scenario={}, agents=[], state={})
            with log.path.open("a") as handle:
                handle.write('{"seq": 2, "type": "turn_')
            self.assertEqual(len(read_events(log.path)), 1)
            with self.assertRaises(ValueError):
                log.emit("bogus")


class LegacySocietyEventLogTests(unittest.TestCase):
    def test_society_run_emits_monitor_compatible_events(self):
        from genesis.simulation import CONFIG, run_simulation

        config = json.loads(json.dumps(CONFIG))
        config["llm"] = {"provider": "mock"}
        with tempfile.TemporaryDirectory() as tmp:
            state = run_simulation(days=2, reset=True, config=config, db_path=f"{tmp}/g.db",
                                   seed=1, events_dir=f"{tmp}/runs")
            events = read_events(f"{tmp}/runs/{state['run_id']}/events.jsonl")
            self.assertEqual(types(events)[0], "run_start")
            self.assertEqual(types(events)[-1], "run_end")
            self.assertEqual(sum(e["type"] == "agent_step" for e in events), 10)
            self.assertEqual(sum(e["type"] == "turn_end" for e in events), 2)
            step = next(e for e in events if e["type"] == "agent_step")
            self.assertIn("reasoning", step)
            self.assertIn("changes", step["effect"])
            self.assertEqual(state["usage"]["calls"], 10)
            self.assertEqual(state["usage"]["cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
