"""Action vocabulary, JSON schema, validation and legal-move generation.

``ActionRules`` is the single source of truth for what an agent may do in a scenario:
the same constraint table produces the schema sent to the model, validates the
model's reply, and enumerates legal moves for the mock provider, so they cannot drift.
All checks run against the *perception* dict the agent was shown.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from genesis.llm.structured import extract_json_object

ACTION_DOCS = {
    "pass": "Do nothing this turn. No parameters.",
    "say": "Say something everyone can read. Needs: message.",
    "whisper": "Send a private message only the target sees. Needs: target, message.",
    "give": "Give your own resource to another agent. Needs: target, resource, amount.",
    "contribute": "Put your own resource into the shared pool, or into a shared project "
                  "if 'project' is set. Needs: resource, amount; optional project.",
    "draw": "Take a resource from the shared pool for yourself (limits apply). "
            "Needs: resource, amount.",
    "propose": "Offer another agent a trade; they may accept next turn. Needs: target, "
               "resource, amount (what you give), request_resource, request_amount (what you want).",
    "accept": "Accept a trade offer addressed to you. Needs: offer_id.",
    "reject": "Reject a trade offer addressed to you. Needs: offer_id.",
    "vote": "Cast (or change) your ballot on an open vote. Needs: vote_id, option.",
    "take": "Try to take a resource from another agent without consent. May fail and be "
            "exposed publicly. Needs: target, resource, amount.",
}
FIELDS = ("target", "resource", "amount", "request_resource", "request_amount",
          "offer_id", "vote_id", "option", "project", "message")
_REQUIRED_BY_ACTION = {
    "say": ("message",), "whisper": ("target", "message"),
    "give": ("target", "resource", "amount"), "contribute": ("resource", "amount"),
    "draw": ("resource", "amount"),
    "propose": ("target", "resource", "amount", "request_resource", "request_amount"),
    "accept": ("offer_id",), "reject": ("offer_id",),
    "vote": ("vote_id", "option"), "take": ("target", "resource", "amount"),
}


def _coerce_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class ActionRules:
    def __init__(self, spec: dict):
        self.spec = spec
        self.actions: List[str] = list(spec["constraints"]["actions"])
        self.resources: List[str] = list(spec["resources"])
        self.constraints = spec["constraints"]

    # -- schema sent to the provider ----------------------------------------
    def json_schema(self) -> dict:
        nullable_str = {"type": ["string", "null"]}
        nullable_int = {"type": ["integer", "null"]}
        props = {
            "reasoning": {"type": "string", "description":
                          "1-3 sentence rationale for this choice (self-reported)."},
            "action": {"type": "string", "enum": self.actions},
            "target": nullable_str, "resource": nullable_str, "amount": nullable_int,
            "request_resource": nullable_str, "request_amount": nullable_int,
            "offer_id": nullable_str, "vote_id": nullable_str, "option": nullable_str,
            "project": nullable_str, "message": nullable_str,
        }
        return {"name": "agent_action", "schema": {
            "type": "object", "properties": props,
            "required": list(props), "additionalProperties": False,
        }}

    def catalog(self) -> Dict[str, str]:
        return {name: ACTION_DOCS[name] for name in self.actions}

    # -- parsing -------------------------------------------------------------
    def parse(self, raw: str, perception: dict) -> Tuple[Optional[dict], List[str]]:
        """Parse a model reply into ``(action, errors)``; ``action`` is None on errors."""
        try:
            data = extract_json_object(raw)
        except (json.JSONDecodeError, TypeError):
            return None, ["reply was not valid JSON"]
        if not isinstance(data, dict):
            return None, ["reply must be a JSON object"]
        return self.check(perception, data)

    def check(self, perception: dict, data: dict) -> Tuple[Optional[dict], List[str]]:
        errors: List[str] = []
        action = data.get("action")
        if action not in self.actions:
            return None, [f"action must be one of: {', '.join(self.actions)}"]
        limits = self.constraints
        you = perception["you"]
        others = {o["id"]: o for o in perception["others"]}
        max_amount = limits["max_amount"]
        out: Dict[str, Any] = {name: None for name in FIELDS}
        out["action"] = action
        out["reasoning"] = (_clean_str(data.get("reasoning")) or "")[:600]

        def need(field):
            if data.get(field) in (None, ""):
                errors.append(f"{action} needs '{field}'")
                return False
            return True

        def resource(field="resource"):
            value = data.get(field)
            if not need(field):
                return None
            if value not in self.resources:
                errors.append(f"{field} '{value}' is not one of: {', '.join(self.resources)}")
                return None
            return value

        def amount(field="amount", cap=max_amount):
            if not need(field):
                return None
            value = _coerce_int(data.get(field))
            if value is None or not 1 <= value <= cap:
                errors.append(f"{field} must be an integer from 1 to {cap}")
                return None
            return value

        def target():
            if not need("target"):
                return None
            value = data.get("target")
            if value == you["id"]:
                errors.append("target cannot be yourself")
                return None
            if value not in others or not others[value].get("alive", True):
                errors.append(f"target '{value}' is not a living agent you can act on")
                return None
            return value

        def message(required):
            text = _clean_str(data.get("message"))
            if required and not text:
                errors.append(f"{action} needs a non-empty 'message'")
            return text[: limits["message_max_chars"]] if text else None

        own = you["resources"]
        if action == "say":
            out["message"] = message(True)
        elif action == "whisper":
            out["target"], out["message"] = target(), message(True)
        elif action == "give":
            out["target"], out["resource"] = target(), resource()
            cap = min(max_amount, own.get(out["resource"], 0)) if out["resource"] else max_amount
            out["amount"] = amount(cap=max(cap, 1))
            if out["resource"] and own.get(out["resource"], 0) < 1:
                errors.append(f"you have no {out['resource']} to give")
        elif action == "contribute":
            out["resource"] = resource()
            cap = min(max_amount, own.get(out["resource"], 0)) if out["resource"] else max_amount
            out["amount"] = amount(cap=max(cap, 1))
            if out["resource"] and own.get(out["resource"], 0) < 1:
                errors.append(f"you have no {out['resource']} to contribute")
            project = _clean_str(data.get("project"))
            if project:
                info = perception["world"]["projects"].get(project)
                if info is None:
                    errors.append(f"unknown project '{project}'")
                elif info["done"]:
                    errors.append(f"project '{project}' is already complete")
                elif out["resource"] and info["remaining"].get(out["resource"], 0) < 1:
                    errors.append(f"project '{project}' needs no more {out['resource']}")
                out["project"] = project
        elif action == "draw":
            out["resource"] = resource()
            cap = max_amount
            if out["resource"]:
                cap = min(max_amount, limits["draw_limit"].get(out["resource"], max_amount))
            if cap < 1:
                errors.append(f"drawing {out['resource']} is not allowed")
            out["amount"] = amount(cap=max(cap, 1))
        elif action == "propose":
            out["target"], out["resource"] = target(), resource()
            cap = min(max_amount, own.get(out["resource"], 0)) if out["resource"] else max_amount
            out["amount"] = amount(cap=max(cap, 1))
            out["request_resource"] = resource("request_resource")
            out["request_amount"] = amount("request_amount")
            if out["resource"] and own.get(out["resource"], 0) < 1:
                errors.append(f"you have no {out['resource']} to offer")
            if out["resource"] and out["resource"] == out["request_resource"]:
                errors.append("offered and requested resources must differ")
            out["message"] = message(False)
        elif action in ("accept", "reject"):
            offer_id = _clean_str(data.get("offer_id"))
            if offer_id not in {o["id"] for o in perception["offers_to_you"]}:
                errors.append(f"offer_id must be one of: "
                              f"{', '.join(o['id'] for o in perception['offers_to_you']) or '(none open)'}")
            out["offer_id"] = offer_id
            out["message"] = message(False)
        elif action == "vote":
            votes = {v["id"]: v for v in perception["open_votes"]}
            vote_id, option = _clean_str(data.get("vote_id")), _clean_str(data.get("option"))
            if vote_id not in votes:
                errors.append(f"vote_id must be one of: {', '.join(votes) or '(no open votes)'}")
            elif option not in votes[vote_id]["options"]:
                errors.append(f"option must be one of: {', '.join(votes[vote_id]['options'])}")
            out["vote_id"], out["option"] = vote_id, option
            out["message"] = message(False)
        elif action == "take":
            out["target"], out["resource"], out["amount"] = target(), resource(), amount()
            out["message"] = message(False)
        else:  # pass
            out["message"] = message(False)
        if errors:
            return None, errors
        return out, []

    # -- legal move enumeration (mock provider, hints, tests) ------------------
    def candidates(self, perception: dict, per_kind: int = 6) -> List[dict]:
        you = perception["you"]
        own = you["resources"]
        living = [o["id"] for o in perception["others"] if o.get("alive", True)]
        max_amount = self.constraints["max_amount"]
        out: List[dict] = []

        def add(item):
            normalized, errors = self.check(perception, {"reasoning": "", **item})
            if not errors:
                out.append({**{k: v for k, v in normalized.items() if v is not None}})

        for action in self.actions:
            before = len(out)
            if action == "pass":
                add({"action": "pass"})
            elif action == "say":
                add({"action": "say", "message": "I will help us stay on track."})
            elif action == "whisper":
                for t in living:
                    add({"action": "whisper", "target": t, "message": "Can we coordinate privately?"})
            elif action in ("give", "take"):
                for t in living:
                    for r in self.resources:
                        for n in range(1, max_amount + 1):
                            add({"action": action, "target": t, "resource": r, "amount": n})
            elif action == "contribute":
                projects = [None] + [p for p, i in perception["world"]["projects"].items() if not i["done"]]
                for r in self.resources:
                    for n in range(1, max_amount + 1):
                        for p in projects:
                            add({"action": "contribute", "resource": r, "amount": n, "project": p})
            elif action == "draw":
                for r in self.resources:
                    for n in range(1, max_amount + 1):
                        add({"action": "draw", "resource": r, "amount": n})
            elif action == "propose":
                for t in living:
                    for r in self.resources:
                        for rr in self.resources:
                            add({"action": "propose", "target": t, "resource": r, "amount": 1,
                                 "request_resource": rr, "request_amount": 1,
                                 "message": "A fair swap?"})
            elif action in ("accept", "reject"):
                for offer in perception["offers_to_you"]:
                    add({"action": action, "offer_id": offer["id"]})
            elif action == "vote":
                for vote in perception["open_votes"]:
                    for option in vote["options"]:
                        add({"action": "vote", "vote_id": vote["id"], "option": option})
            # cap each verb so large worlds stay cheap, but keep variety
            kind = out[before:]
            if len(kind) > per_kind:
                step = len(kind) / per_kind
                out[before:] = [kind[int(i * step)] for i in range(per_kind)]
        return out
