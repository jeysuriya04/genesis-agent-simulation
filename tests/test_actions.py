import json
import unittest

from genesis.actions import ActionRules
from genesis.engine import build_perception, initial_world
from tests.helpers import tiny_spec


class ActionParsingTests(unittest.TestCase):
    def setUp(self):
        self.spec = tiny_spec()
        self.rules = ActionRules(self.spec)
        self.world = initial_world(self.spec)
        self.world["turn"] = 1
        self.world["offers"].append({
            "id": "O0-1", "from": "b", "to": "a", "give_resource": "water", "give_amount": 1,
            "want_resource": "labor", "want_amount": 1, "turn": 0, "status": "open", "expires_turn": 3})
        self.view = build_perception(self.world, self.spec, self.rules, "a", [])

    def parse(self, **fields):
        return self.rules.check(self.view, {"reasoning": "r", **fields})

    def assert_error(self, text, **fields):
        action, errors = self.parse(**fields)
        self.assertIsNone(action)
        self.assertTrue(any(text in e for e in errors), errors)

    def test_valid_actions_for_every_verb(self):
        ok = [
            dict(action="pass"), dict(action="say", message="hi"),
            dict(action="whisper", target="b", message="psst"),
            dict(action="give", target="b", resource="water", amount=2),
            dict(action="contribute", resource="labor", amount=2, project="well"),
            dict(action="contribute", resource="water", amount=1),
            dict(action="draw", resource="water", amount=1),
            dict(action="propose", target="b", resource="water", amount=1,
                 request_resource="labor", request_amount=2),
            dict(action="accept", offer_id="O0-1"), dict(action="reject", offer_id="O0-1"),
            dict(action="vote", vote_id="v1", option="yes"),
            dict(action="take", target="b", resource="water", amount=1),
        ]
        for fields in ok:
            action, errors = self.parse(**fields)
            self.assertEqual(errors, [], fields)
            self.assertEqual(action["action"], fields["action"])

    def test_amount_bounds_and_types(self):
        self.assert_error("amount must be", action="give", target="b", resource="water", amount=0)
        self.assert_error("amount must be", action="give", target="b", resource="water", amount=3)  # > max_amount
        self.assert_error("amount must be", action="give", target="b", resource="water", amount="lots")
        self.assert_error("amount must be", action="give", target="b", resource="water", amount=True)
        action, _ = self.parse(action="give", target="b", resource="water", amount="2")  # numeric string ok
        self.assertEqual(action["amount"], 2)
        action, _ = self.parse(action="give", target="b", resource="water", amount=2.0)
        self.assertEqual(action["amount"], 2)

    def test_draw_limit_and_resource_names(self):
        self.assert_error("amount must be", action="draw", resource="water", amount=2)  # limit 1
        self.assert_error("not allowed", action="draw", resource="labor", amount=1)  # limit 0
        self.assert_error("is not one of", action="draw", resource="gold", amount=1)

    def test_cannot_spend_more_than_you_have_or_target_invalid_agents(self):
        self.world["agents"]["a"]["resources"]["water"] = 0
        view = build_perception(self.world, self.spec, self.rules, "a", [])
        _, errors = self.rules.check(view, {"action": "give", "target": "b", "resource": "water", "amount": 1})
        self.assertTrue(any("no water" in e for e in errors))
        self.assert_error("not a living agent", action="give", target="zed", resource="labor", amount=1)
        self.assert_error("yourself", action="whisper", target="a", message="x")

    def test_required_fields_ids_and_options(self):
        self.assert_error("needs a non-empty 'message'", action="say", message="   ")
        self.assert_error("offer_id must be", action="accept", offer_id="nope")
        self.assert_error("vote_id must be", action="vote", vote_id="zzz", option="yes")
        self.assert_error("option must be", action="vote", vote_id="v1", option="maybe")
        self.assert_error("unknown project", action="contribute", resource="labor", amount=1, project="moon")
        self.assert_error("must differ", action="propose", target="b", resource="water", amount=1,
                          request_resource="water", request_amount=1)
        self.assert_error("action must be one of", action="fly")

    def test_raw_parsing_handles_fences_garbage_and_non_objects(self):
        raw = "```json\n" + json.dumps({"action": "say", "message": "hello", "reasoning": "x"}) + "\n```"
        action, errors = self.rules.parse(raw, self.view)
        self.assertEqual((errors, action["message"]), ([], "hello"))
        self.assertEqual(self.rules.parse("no json here", self.view)[1], ["reply was not valid JSON"])
        self.assertEqual(self.rules.parse("[1,2]", self.view)[1], ["reply must be a JSON object"])

    def test_message_is_truncated_to_the_scenario_limit(self):
        action, _ = self.parse(action="say", message="x" * 1000)
        self.assertEqual(len(action["message"]), self.spec["constraints"]["message_max_chars"])

    def test_candidates_are_all_valid_and_cover_allowed_verbs(self):
        candidates = self.rules.candidates(self.view)
        self.assertTrue(candidates)
        for candidate in candidates:
            action, errors = self.rules.check(self.view, {"reasoning": "", **candidate})
            self.assertEqual(errors, [], candidate)
        verbs = {c["action"] for c in candidates}
        self.assertEqual(verbs, set(self.spec["constraints"]["actions"]))

    def test_schema_is_strict_with_every_property_required(self):
        schema = self.rules.json_schema()["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(schema["properties"]["action"]["enum"], self.spec["constraints"]["actions"])


if __name__ == "__main__":
    unittest.main()
