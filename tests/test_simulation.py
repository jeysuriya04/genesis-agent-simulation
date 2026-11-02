import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from genesis.scenarios import load_scenarios
from genesis.simulation import (
    initial_state,
    run_simulation,
    state_summary,
    validate_decision,
)
from genesis.storage import connect


BASE = Path(__file__).resolve().parents[1]


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((BASE / "config.json").read_text())
        self.config["days"] = 1
        self.config["starting_food"] = 3
        self.config["starting_medicine"] = 1
        self.config["daily_medicine_supply"] = 0
        self.config["daily_medicine_consumption"] = 0

    def test_decision_validation_rejects_unknown_targets_and_excess_amounts(self):
        state = initial_state(self.config)
        context = state_summary(state, "A2")
        decision = validate_decision(
            json.dumps({"action": "take", "target": "unknown", "amount": 99}),
            context,
        )
        self.assertEqual(decision["action"], "observe")
        self.assertIn("INVALID_TARGET", decision["validation_flags"])
        self.assertIn("INVALID_AMOUNT", decision["validation_flags"])
        malformed = validate_decision(
            json.dumps(
                {
                    "action": "take",
                    "target": [],
                    "amount": {"unexpected": "object"},
                    "related_event_id": [],
                }
            ),
            context,
        )
        self.assertEqual(malformed["action"], "observe")
        self.assertIn("INVALID_TARGET", malformed["validation_flags"])
        self.assertIn("INVALID_AMOUNT", malformed["validation_flags"])

    def test_agent_prompt_history_is_bounded_and_world_events_are_summarized(self):
        state = initial_state(self.config)
        state["history"] = [
            {
                "id": index,
                "day": index,
                "agent_id": "A2",
                "agent_name": "Bala",
                "action": "observe",
                "target": None,
                "message": "I observe the settlement.",
                "outcome": "No material resource change.",
                "ethical_flags": [],
                "scenario_id": None,
            }
            for index in range(1, 21)
        ]
        state["world_events"] = [
            {
                "day": index,
                "scenario_id": f"scenario-{index}",
                "title": f"Scenario {index}",
                "daily_resource_report": {
                    "unmet_food_need": ["A1"],
                    "unmet_medicine_need": [],
                    "world_after": {
                        "community": {"food": 3, "medicine": 2},
                        "agents": {"very_large_history_payload": "x" * 10000},
                    },
                },
            }
            for index in range(1, 21)
        ]
        context = state_summary(state, "A1")
        self.assertEqual(len(context["recent_events"]), 6)
        self.assertEqual(len(context["recent_world_events"]), 3)
        self.assertEqual(len(context["personal_memories"]), 0)
        self.assertNotIn(
            "agents", context["recent_world_events"][0]["community_resources"]
        )
        self.assertLess(len(json.dumps(context)), 10000)

    def test_virtual_days_concurrent_snapshot_consent_trade_and_event_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "genesis.db"
            day_one_contexts = {}
            resolved = []

            def provider(agent, context):
                if context["day"] == 1:
                    day_one_contexts[agent["id"]] = context["verifiable_facts"]
                if context["day"] == 1 and agent["id"] == "A1":
                    decision = {
                        "action": "propose_trade",
                        "target": "A2",
                        "amount": 1,
                        "offered_resource": "food",
                        "requested_resource": "medicine",
                        "requested_amount": 1,
                        "message": "One food for one medicine.",
                    }
                elif context["day"] == 2 and agent["id"] == "A2":
                    offer = context["pending_trade_offers"][0]
                    decision = {
                        "action": "accept_trade",
                        "trade_offer_id": offer["id"],
                        "message": "I accept the offer.",
                    }
                elif context["day"] == 1 and agent["id"] == "A3":
                    decision = {
                        "action": "share",
                        "target": "A4",
                        "resource": "food",
                        "amount": 1,
                        "message": "Sharing one unit.",
                    }
                else:
                    decision = {"action": "observe", "message": "I observe."}
                return {
                    "raw": json.dumps(decision),
                    "model_version": "test-model",
                    "prompt_json": {
                        "system": "test system prompt",
                        "user": "test user context",
                    },
                }

            def progress(day, agent_id, event):
                resolved.append((day, agent_id, event))

            run_simulation(
                days=1,
                reset=True,
                seed=7,
                concurrency=5,
                decision_provider=provider,
                config=self.config,
                db_path=db_path,
                progress=progress,
            )
            # Resume the same experiment into day two; its offer is now visible to the recipient.
            state = run_simulation(
                days=2,
                decision_provider=provider,
                config=self.config,
                db_path=db_path,
                progress=progress,
            )

            self.assertEqual(state["day"], 3)
            self.assertEqual(len(day_one_contexts), 5)
            self.assertEqual(len({json.dumps(value, sort_keys=True) for value in day_one_contexts.values()}), 1)
            self.assertEqual(len(resolved), 10)
            self.assertEqual(state["pending_trades"][0]["status"], "accepted")
            self.assertEqual(state["agents"]["A1"]["food"], 2)
            self.assertEqual(state["agents"]["A2"]["medicine"], 0)
            self.assertEqual(state["agents"]["A2"]["food"], 4)

            con = sqlite3.connect(db_path)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM events").fetchone()[0], 12)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM events WHERE scenario_id = 'opening-council'"
                ).fetchone()[0],
                6,
            )
            row = con.execute(
                "SELECT latency_ms, model_version, prompt_hash, world_before_json, "
                "consequences_json, behavior_json, prompt_json FROM events "
                "WHERE agent_id != 'WORLD' ORDER BY id LIMIT 1"
            ).fetchone()
            self.assertGreaterEqual(row[0], 0)
            self.assertEqual(row[1], "test-model")
            self.assertTrue(row[2])
            self.assertTrue(json.loads(row[3]))
            self.assertIn("relationships", json.loads(row[3]))
            self.assertTrue(json.loads(row[4]))
            self.assertIn("cooperation", json.loads(row[5]))
            self.assertEqual(
                json.loads(row[6])["system"], "test system prompt"
            )
            run_scenarios = con.execute(
                "SELECT scenario_json FROM runs LIMIT 1"
            ).fetchone()[0]
            self.assertEqual(json.loads(run_scenarios)["version"], "1.0")
            con.close()

    def test_failed_model_call_is_persisted_as_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "genesis.db"

            def fail_provider(agent, context):
                raise TimeoutError("simulated timeout")

            state = run_simulation(
                days=1,
                reset=True,
                decision_provider=fail_provider,
                config=self.config,
                db_path=db_path,
            )
            self.assertEqual(state["run_status"], "completed_with_errors")
            con = sqlite3.connect(db_path)
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM events WHERE api_error IS NOT NULL").fetchone()[0],
                5,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM events WHERE action = 'observe' "
                    "AND ethical_flags LIKE '%API_REQUEST_FAILED%'"
                ).fetchone()[0],
                5,
            )
            con.close()

    def test_schema_migration_preserves_legacy_events_and_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "legacy.db"
            con = sqlite3.connect(db_path)
            con.executescript(
                """
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT,
                    amount INTEGER DEFAULT 0,
                    message TEXT,
                    ethical_flags TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    model_output TEXT NOT NULL,
                    state_after_json TEXT NOT NULL
                );
                CREATE TABLE snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    state_json TEXT NOT NULL
                );
                INSERT INTO events VALUES (
                    1, '2026-01-01T00:00:00Z', 1, 'A1', 'Asha', 'old-model',
                    'observe', NULL, 0, 'old record', '[]', '{}', '{}', '{}'
                );
                INSERT INTO snapshots VALUES (1, '2026-01-01T00:00:00Z', 1, '{}');
                """
            )
            con.close()

            migrated = connect(db_path)
            event = migrated.execute(
                "SELECT message, run_id, prompt_hash FROM events WHERE id = 1"
            ).fetchone()
            self.assertEqual(event["message"], "old record")
            self.assertEqual(event["run_id"], "legacy")
            self.assertIsNone(event["prompt_hash"])
            self.assertEqual(
                migrated.execute("SELECT COUNT(*) FROM runs WHERE run_id = 'legacy'").fetchone()[0],
                1,
            )
            migrated.close()

    def test_scenario_configuration_is_loadable_and_validated(self):
        scenarios = load_scenarios(BASE / "scenarios.json")
        self.assertEqual(scenarios["version"], "1.0")
        self.assertEqual(scenarios["scenarios"][0]["id"], "opening-council")

    def test_virtual_day_resource_shortage_and_world_event_are_persisted(self):
        config = json.loads((BASE / "config.json").read_text())
        config["days"] = 11
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "shortage.db"
            state = run_simulation(
                days=11,
                reset=True,
                decision_provider=lambda agent, context: {
                    "action": "observe",
                    "message": "I observe the situation.",
                },
                config=config,
                db_path=db_path,
            )
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            report = con.execute(
                "SELECT context_json, consequences_json FROM events "
                "WHERE agent_id = 'WORLD' AND day = 11"
            ).fetchone()
            self.assertIsNotNone(report)
            report_context = json.loads(report["context_json"])
            daily = report_context["daily_resource_report"]
            self.assertEqual(daily["medicine_supply_per_agent"], 0)
            self.assertEqual(len(daily["unmet_medicine_need"]), 4)
            self.assertEqual(len(daily["health_after"]), 5)
            self.assertEqual(len(daily["per_agent"]), 5)
            self.assertEqual(
                sum(
                    entry["medicine_consumed_from_community"]
                    for entry in daily["per_agent"].values()
                ),
                1,
            )
            self.assertEqual(state["day"], 12)
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM events WHERE agent_id != 'WORLD'"
                ).fetchone()[0],
                55,
            )
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
                11,
            )
            con.close()

    def test_rule_proposal_requires_majority_and_becomes_public_rule(self):
        state = initial_state(self.config, run_id="testrun")
        context = state_summary(state, "A1")
        proposal = validate_decision(
            json.dumps(
                {"action": "govern", "message": "Publish shared-store accounts."}
            ),
            context,
        )
        from genesis.simulation import apply_action

        apply_action(state, "A1", proposal, context, 1)
        proposal_id = state["proposals"][0]["id"]
        self.assertEqual(state["proposals"][0]["status"], "open")
        for agent_id in ("A2", "A3"):
            vote_context = state_summary(state, agent_id)
            vote = validate_decision(
                json.dumps(
                    {"action": "vote_for_rule", "proposal_id": proposal_id}
                ),
                vote_context,
            )
            apply_action(state, agent_id, vote, vote_context, 2)
        self.assertEqual(state["proposals"][0]["status"], "enacted")
        self.assertIn("Publish shared-store accounts.", state["active_rules"])
        self.assertIn(
            "Publish shared-store accounts.",
            state_summary(state, "A4")["public_rules"],
        )

    def test_forgiveness_requires_and_resolves_recipient_open_apology(self):
        from genesis.simulation import apply_action

        state = initial_state(self.config, run_id="testrun")
        apology_context = state_summary(state, "A1")
        apology = validate_decision(
            json.dumps(
                {"action": "apologize", "target": "A2", "message": "I regret taking it."}
            ),
            apology_context,
        )
        apply_action(state, "A1", apology, apology_context, 1)

        recipient_context = state_summary(state, "A2")
        forgive = validate_decision(
            json.dumps({"action": "forgive", "target": "A1"}),
            recipient_context,
        )
        self.assertEqual(forgive["action"], "forgive")
        apply_action(state, "A2", forgive, recipient_context, 2)
        self.assertEqual(state["apologies"][0]["status"], "forgiven")
        self.assertGreater(state["relationships"]["A2"]["A1"]["trust"], 0)
        repeated = validate_decision(
            json.dumps({"action": "forgive", "target": "A1"}),
            state_summary(state, "A2"),
        )
        self.assertEqual(repeated["action"], "observe")
        self.assertIn("NO_OPEN_APOLOGY", repeated["validation_flags"])

    def test_mocked_30_day_experiment_records_five_decisions_each_day(self):
        config = json.loads((BASE / "config.json").read_text())

        def provider(agent, context):
            if context["day"] == 26 and agent["id"] == "A1":
                return {
                    "action": "govern",
                    "message": "Publish a daily community resource ledger.",
                }
            if context["day"] == 27 and agent["id"] in {"A2", "A3"}:
                proposal = context["open_proposals"][0]
                return {
                    "action": "vote_for_rule",
                    "proposal_id": proposal["id"],
                }
            return {"action": "observe", "message": "I observe the situation."}

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "thirty-days.db"
            state = run_simulation(
                days=30,
                reset=True,
                seed=19,
                decision_provider=provider,
                config=config,
                db_path=db_path,
            )
            con = sqlite3.connect(db_path)
            self.assertEqual(state["day"], 31)
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM events WHERE agent_id != 'WORLD'"
                ).fetchone()[0],
                150,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM events WHERE agent_id = 'WORLD'"
                ).fetchone()[0],
                30,
            )
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
                30,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM events WHERE scenario_id IS NOT NULL"
                ).fetchone()[0],
                66,
            )
            con.close()
            self.assertEqual(
                state["proposals"][0]["status"],
                "enacted",
            )

    def test_reset_creates_new_run_without_deleting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "runs.db"
            provider = lambda agent, context: {"action": "observe"}
            run_simulation(
                days=1, reset=True, decision_provider=provider,
                config=self.config, db_path=db_path,
            )
            run_simulation(
                days=1, reset=True, decision_provider=provider,
                config=self.config, db_path=db_path,
            )
            con = sqlite3.connect(db_path)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM events").fetchone()[0], 12)
            con.close()

if __name__ == "__main__":
    unittest.main()
