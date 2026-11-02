import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import showcase
from genesis.simulation import run_simulation


BASE = Path(__file__).resolve().parents[1]


class ShowcaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "showcase.db"
        self.config = json.loads((BASE / "config.json").read_text())
        self.config["days"] = 2
        self.config["starting_medicine"] = 1
        self.config["daily_medicine_supply"] = 0
        self.config["daily_medicine_consumption"] = 0

        def provider(agent, context):
            if context["day"] == 1 and agent["id"] == "A1":
                return {
                    "action": "share",
                    "target": "A2",
                    "amount": 1,
                    "message": "I share one food.",
                    "reason": "Bala needs help.",
                }
            return {"action": "observe", "message": "I observe."}

        self.state = run_simulation(
            days=2,
            reset=True,
            seed=5,
            decision_provider=provider,
            config=self.config,
            db_path=self.db_path,
        )
        self.run_id = self.state["run_id"]

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_run_replay_loads_timeline_profiles_charts_and_relationships(self):
        with patch.object(showcase, "DB_PATH", self.db_path):
            outputs = showcase.load_showcase(
                self.run_id, 0, "All residents", "trust"
            )

        self.assertEqual(len(outputs), 13)
        overview, timeline, actions, resources, behavior, propagation = outputs[:6]
        self.assertIn("10/10", overview)
        self.assertEqual(len(timeline), 12)
        self.assertEqual(timeline.iloc[0]["Resident"], "World")
        self.assertIsNotNone(actions)
        self.assertIsNotNone(resources)
        self.assertIsNotNone(behavior)
        self.assertIsInstance(propagation, pd.DataFrame)
        self.assertIn("Asha", outputs[8])
        self.assertGreaterEqual(len(outputs[9].data), 1)
        self.assertEqual(len(outputs[10]), 0)
        self.assertEqual(outputs[11]["action"], "environment_update")

    def test_day_and_resident_filters_and_event_details(self):
        with patch.object(showcase, "DB_PATH", self.db_path):
            outputs = showcase.load_showcase(
                self.run_id, 1, "A1", "friendship"
            )
            timeline = outputs[1]
            event_choices = outputs[7].choices
            selected_event = event_choices[0][1]
            details, notes, status = showcase.load_event_detail(
                self.run_id, selected_event
            )

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline.iloc[0]["Resident"], "Asha")
        self.assertEqual(details["event_id"], selected_event)
        self.assertIn("available_context", details)
        self.assertEqual(status, "unreviewed")
        self.assertEqual(notes, "")

    def test_human_review_persists_to_selected_run_event(self):
        with patch.object(showcase, "DB_PATH", self.db_path):
            choices = showcase.load_showcase(
                self.run_id, 1, "A1", "trust"
            )[7].choices
            event_id = choices[0][1]
            message = showcase.save_human_review(
                self.run_id, event_id, "reviewed", "Reviewed the event context."
            )

        self.assertIn("saved", message)
        db = sqlite3.connect(self.db_path)
        row = db.execute(
            "SELECT review_status, review_notes FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        db.close()
        self.assertEqual(
            row, ("reviewed", "Reviewed the event context.")
        )

    def test_gradio_application_builds(self):
        with patch.object(showcase, "DB_PATH", self.db_path):
            app = showcase.build_app()
        config = app.get_config_file()
        run_ledger = next(
            component
            for component in config["components"]
            if component.get("props", {}).get("label") == "Available experiments"
        )
        self.assertEqual(
            run_ledger["props"]["value"]["headers"],
            ["Run", "Run ID"],
        )


if __name__ == "__main__":
    unittest.main()
