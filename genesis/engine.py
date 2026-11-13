"""Turn-based scenario engine.

Mirrors the original simulator's design: every agent decides from the *same*
start-of-turn snapshot (concurrently), then the engine validates and resolves the
decisions one by one in a seeded order, so the engine - not the model - owns the world.
Every step is written to the JSONL event log for monitoring and replay.
"""
from __future__ import annotations

import copy
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from genesis.actions import ActionRules
from genesis.events import EventLog
from genesis.llm import (
    LLMAuthError, LLMBadRequestError, LLMError, LLMRequest, ProviderPool, UsageTracker, request_structured,
)
from genesis.scenario_spec import OPS

RECENT_LOG = 10


# --------------------------------------------------------------------------- world
def initial_world(spec: dict) -> dict:
    agents = {}
    for agent in spec["agents"]:
        resources = {
            name: agent["start"].get(name, info.get("agent_start", 0))
            for name, info in spec["resources"].items()
        }
        agents[agent["id"]] = {"resources": resources, "health": spec["health"]["initial"],
                               "alive": True}
    return {
        "turn": 0,
        "pool": {name: info.get("pool", 0) for name, info in spec["resources"].items()},
        "variables": {name: info["initial"] for name, info in spec["variables"].items()},
        "agents": agents,
        "projects": {
            pid: {"progress": {res: 0 for res in info["needs"]}, "done": False}
            for pid, info in spec["projects"].items()
        },
        "offers": [],
        "votes": {v["id"]: {"ballots": {}, "result": None} for v in spec["votes"]},
        "public_log": [],
        "private_messages": [],
    }


def public_state(world: dict) -> dict:
    """The state shown on the dashboard (no logs, which are carried by agent steps)."""
    return {
        "turn": world["turn"],
        "pool": dict(world["pool"]),
        "variables": dict(world["variables"]),
        "agents": copy.deepcopy(world["agents"]),
        "projects": copy.deepcopy(world["projects"]),
        "offers": [o for o in world["offers"] if o["status"] == "open"],
        "votes": {k: {"ballots": dict(v["ballots"]), "result": v["result"]}
                  for k, v in world["votes"].items()},
    }


def flatten(world: dict, spec: dict) -> Dict[str, float]:
    """Metric path -> value, used both for diffs and for success/failure conditions."""
    flat: Dict[str, float] = {"turn": world["turn"]}
    for res, value in world["pool"].items():
        flat[f"pool.{res}"] = value
    for name, value in world["variables"].items():
        flat[f"var.{name}"] = value
    living = [a for a in world["agents"].values() if a["alive"]]
    flat["agents.alive"] = len(living)
    for aid, agent in world["agents"].items():
        flat[f"agent.{aid}.health"] = agent["health"]
        for res, value in agent["resources"].items():
            flat[f"agent.{aid}.{res}"] = value
    for field in list(spec["resources"]) + ["health"]:
        series = [(a["health"] if field == "health" else a["resources"][field]) for a in living] or [0]
        flat[f"agents.min.{field}"] = min(series)
        flat[f"agents.max.{field}"] = max(series)
        flat[f"agents.sum.{field}"] = sum(series)
    for pid, project in world["projects"].items():
        flat[f"project.{pid}.progress"] = sum(project["progress"].values())
        flat[f"project.{pid}.done"] = 1 if project["done"] else 0
    scores = scores_for(world, spec)
    for aid, score in scores.items():
        flat[f"score.{aid}"] = score
    if scores:
        flat["score.max"], flat["score.min"] = max(scores.values()), min(scores.values())
    return flat


def scores_for(world: dict, spec: dict) -> Dict[str, float]:
    scoring = spec.get("scoring")
    if not scoring:
        return {}
    return {
        aid: round(sum(w * agent["resources"].get(res, 0) for res, w in scoring["weights"].items()), 2)
        for aid, agent in world["agents"].items()
    }


def diff(before: Dict[str, float], after: Dict[str, float]) -> List[dict]:
    skip = ("agents.", "score.", "turn")
    return [
        {"path": path, "before": before.get(path), "after": value}
        for path, value in after.items()
        if before.get(path) != value and not path.startswith(skip)
    ]


def apply_effect(world: dict, spec: dict, effect: dict) -> None:
    op, value, target = effect["op"], effect["value"], effect["target"]
    parts = target.split(".")

    def update(container, key, floor=0):
        container[key] = max(floor, container[key] + value if op == "add" else value)

    if parts[0] == "pool":
        update(world["pool"], parts[1])
    elif parts[0] == "var":
        update(world["variables"], parts[1], floor=-10**9)
    else:
        if parts[0] == "agent":
            ids = [parts[1]]
        else:
            ids = [a for a, v in world["agents"].items() if parts[1] == "all" or v["alive"]]
        for aid in ids:
            agent = world["agents"][aid]
            if parts[2] == "health":
                update(agent, "health")
                agent["health"] = min(agent["health"], spec["health"]["initial"])
                if agent["health"] <= 0:
                    kill(world, aid)
            else:
                update(agent["resources"], parts[2])


def kill(world: dict, aid: str) -> None:
    agent = world["agents"][aid]
    if not agent["alive"]:
        return
    agent["alive"], agent["health"] = False, 0
    for res, amount in agent["resources"].items():  # the estate returns to the commons
        world["pool"][res] += amount
        agent["resources"][res] = 0


def evaluate_group(group: Optional[dict], flat: dict, turn: int, limit: int) -> Tuple[bool, List[dict]]:
    """Return (met, per-condition details). 'at_limit' conditions count only on the last turn."""
    if not group:
        return False, []
    details = []
    for mode in ("all", "any"):
        for cond in group.get(mode, []):
            value = flat.get(cond["metric"], 0)
            live = cond.get("when", "any_turn") == "any_turn" or turn >= limit
            details.append({"id": cond["id"], "mode": mode, "description": cond["description"],
                            "metric": cond["metric"], "op": cond["op"], "target": cond["value"],
                            "value": value, "met": bool(live and OPS[cond["op"]](value, cond["value"]))})
    all_ok = all(d["met"] for d in details if d["mode"] == "all")
    any_list = [d for d in details if d["mode"] == "any"]
    any_ok = any(d["met"] for d in any_list) if any_list else True
    return (all_ok and any_ok), details


# ----------------------------------------------------------------------- perception
def describe_conditions(group: Optional[dict]) -> Optional[dict]:
    if not group:
        return None
    return {"description": group["description"],
            "all_of": [c["description"] for c in group.get("all", [])],
            "any_of": [c["description"] for c in group.get("any", [])]}


def build_perception(world: dict, spec: dict, rules: ActionRules, aid: str,
                     events_now: List[dict]) -> dict:
    definition = next(a for a in spec["agents"] if a["id"] == aid)
    cons = spec["constraints"]
    you = world["agents"][aid]
    turn, limit = world["turn"], spec["turn_limit"]

    def other_view(oid, other):
        view = {"id": oid, "name": _name(spec, oid), "role": _agent(spec, oid)["role"],
                "alive": other["alive"], "health": other["health"]}
        if cons["agent_resources_visible"]:
            view["resources"] = dict(other["resources"])
        return view

    projects = {}
    for pid, info in spec["projects"].items():
        state = world["projects"][pid]
        projects[pid] = {
            "description": info["description"], "needs": info["needs"],
            "progress": dict(state["progress"]), "done": state["done"],
            "remaining": {r: max(0, n - state["progress"][r]) for r, n in info["needs"].items()},
        }
    open_votes = []
    for vote in spec["votes"]:
        if vote["turn"] <= turn <= vote["turn"] + vote.get("window", 1) - 1 \
                and world["votes"][vote["id"]]["result"] is None:
            open_votes.append({
                "id": vote["id"], "question": vote["question"],
                "options": {k: v["description"] for k, v in vote["options"].items()},
                "closes_after_turn": vote["turn"] + vote.get("window", 1) - 1,
                "your_current_ballot": world["votes"][vote["id"]]["ballots"].get(aid),
                "ballots_cast": len(world["votes"][vote["id"]]["ballots"]),
            })
    world_view = {
        "variables": dict(world["variables"]), "projects": projects,
        "events_this_turn": [{"title": e["title"], "description": e["description"]} for e in events_now],
    }
    if cons["pool_visible"]:
        world_view["pool"] = dict(world["pool"])
    perception = {
        "scenario": {"title": spec["title"], "goal": spec["goal"], "briefing": spec["briefing"]},
        "turn": turn, "turn_limit": limit, "turns_remaining_after_this": limit - turn,
        "you": {
            "id": aid, "name": definition["name"], "role": definition["role"],
            "persona": definition["persona"], "goals": definition["goals"],
            "resources": dict(you["resources"]), "health": you["health"], "alive": you["alive"],
            **({"secret_briefing": definition["secret"]} if definition.get("secret") else {}),
        },
        "others": [other_view(o, v) for o, v in world["agents"].items() if o != aid],
        "world": world_view,
        "offers_to_you": [
            {k: o[k] for k in ("id", "from", "give_resource", "give_amount",
                               "want_resource", "want_amount", "expires_turn")}
            for o in world["offers"] if o["to"] == aid and o["status"] == "open"
        ],
        "open_votes": open_votes,
        "private_messages_to_you": [m for m in world["private_messages"]
                                    if m["to"] == aid and m["turn"] >= turn - 2][-5:],
        "recent_public_log": world["public_log"][-RECENT_LOG:],
        "rules": spec["rules"],
        "how_the_scenario_ends": {
            "success": describe_conditions(spec.get("success")),
            "failure": describe_conditions(spec.get("failure")),
            "scoring": spec.get("scoring"),
        },
        "limits": {"max_amount_per_action": cons["max_amount"],
                   "draw_limit_per_turn": cons["draw_limit"],
                   "message_max_chars": cons["message_max_chars"]},
        "allowed_actions": rules.catalog(),
    }
    return perception


def _agent(spec, aid):
    return next(a for a in spec["agents"] if a["id"] == aid)


def _name(spec, aid):
    return _agent(spec, aid)["name"]


SYSTEM_PROMPT = (
    "You are {name}, role: {role}, an agent in a closed multi-agent simulation titled "
    "\"{title}\". Act as this specific person: pursue your goals, respect your persona, and "
    "obey the rules. All agents decide at the same time from the same snapshot, then actions are "
    "resolved in random order, so another agent may act before you. Rules are enforced by the "
    "engine; invalid actions are rejected.\n"
    "Reply with ONE JSON object and nothing else, with exactly these keys: reasoning, action, "
    "target, resource, amount, request_resource, request_amount, offer_id, vote_id, option, "
    "project, message. Use null for keys the action does not need. 'reasoning' is a brief "
    "(1-3 sentences) rationale that researchers will read; it is your own explanation, not proof "
    "of internal thought. Choose exactly one action from allowed_actions. Never state that an "
    "action has already succeeded. Information marked secret_briefing is private to you."
)


# ------------------------------------------------------------------------- resolve
class Resolver:
    """Applies one validated action to the world and reports its effect."""

    def __init__(self, world: dict, spec: dict, seed):
        self.world, self.spec, self.seed = world, spec, seed
        self.draws: Dict[Tuple[str, str], int] = {}
        self.offer_counter = 0

    def say(self, text: str, aid: str, kind="action") -> None:
        self.world["public_log"].append({"turn": self.world["turn"], "agent_id": aid,
                                         "agent": _name(self.spec, aid), "kind": kind, "text": text})

    def resolve(self, aid: str, action: dict) -> dict:
        world, spec = self.world, self.spec
        before = flatten(world, spec)
        name = _name(spec, aid)
        res, amt, target = action["resource"], action["amount"], action["target"]
        ok, flags, interactions = True, [], []
        kind = action["action"]
        summary = f"{name} passed."
        actor = world["agents"][aid]

        def fail(flag, text):
            nonlocal ok, summary
            ok = False
            flags.append(flag)
            summary = text

        if kind == "pass":
            pass
        elif kind == "say":
            summary = f"{name} said: \"{action['message']}\""
            self.say(summary, aid, "say")
        elif kind == "whisper":
            world["private_messages"].append({"turn": world["turn"], "from": aid, "to": target,
                                              "from_name": name, "message": action["message"]})
            summary = f"{name} privately messaged {_name(spec, target)}."
            interactions.append({"from": aid, "to": target, "type": "whisper"})
            self.say(f"{name} whispered something to {_name(spec, target)}.", aid, "whisper")
        elif kind == "give":
            if actor["resources"][res] < amt or not world["agents"][target]["alive"]:
                fail("INSUFFICIENT_RESOURCES", f"{name} could not give {amt} {res} to {_name(spec, target)}.")
            else:
                actor["resources"][res] -= amt
                world["agents"][target]["resources"][res] += amt
                summary = f"{name} gave {amt} {res} to {_name(spec, target)}."
                interactions.append({"from": aid, "to": target, "type": "give", "resource": res, "amount": amt})
                self.say(summary, aid)
        elif kind == "contribute":
            project = action["project"]
            give = min(amt, actor["resources"][res])
            if project:
                remaining = spec["projects"][project]["needs"][res] - world["projects"][project]["progress"][res]
                give = min(give, max(0, remaining))
            if give < 1 or (project and world["projects"][project]["done"]):
                fail("NOTHING_TO_CONTRIBUTE", f"{name} had nothing useful to contribute.")
            else:
                actor["resources"][res] -= give
                if project:
                    world["projects"][project]["progress"][res] += give
                    summary = f"{name} put {give} {res} into project {project}."
                    self._maybe_complete(project, aid)
                else:
                    world["pool"][res] += give
                    summary = f"{name} contributed {give} {res} to the shared pool."
                self.say(summary, aid)
        elif kind == "draw":
            limit = spec["constraints"]["draw_limit"].get(res, spec["constraints"]["max_amount"])
            used = self.draws.get((aid, res), 0)
            got = min(amt, limit - used, world["pool"][res])
            if got < 1:
                fail("POOL_UNAVAILABLE", f"{name} could not draw {res} (limit reached or pool empty).")
            else:
                world["pool"][res] -= got
                actor["resources"][res] += got
                self.draws[(aid, res)] = used + got
                short = f" (asked for {amt})" if got < amt else ""
                summary = f"{name} drew {got} {res} from the shared pool{short}."
                self.say(summary, aid)
        elif kind == "propose":
            self.offer_counter += 1
            offer = {"id": f"O{world['turn']}-{self.offer_counter}", "from": aid, "to": target,
                     "give_resource": res, "give_amount": amt,
                     "want_resource": action["request_resource"], "want_amount": action["request_amount"],
                     "turn": world["turn"], "status": "open",
                     "expires_turn": world["turn"] + spec["constraints"]["offer_expiry_turns"]}
            world["offers"].append(offer)
            summary = (f"{name} offered {_name(spec, target)} {amt} {res} for "
                       f"{offer['want_amount']} {offer['want_resource']} (offer {offer['id']}).")
            interactions.append({"from": aid, "to": target, "type": "propose", "resource": res, "amount": amt})
            self.say(summary, aid)
        elif kind in ("accept", "reject"):
            offer = next((o for o in world["offers"] if o["id"] == action["offer_id"]), None)
            if offer is None or offer["status"] != "open":
                fail("OFFER_CLOSED", f"{name} tried to {kind} an offer that is no longer open.")
            elif kind == "reject":
                offer["status"] = "rejected"
                summary = f"{name} rejected offer {offer['id']} from {_name(spec, offer['from'])}."
                interactions.append({"from": aid, "to": offer["from"], "type": "reject"})
                self.say(summary, aid)
            else:
                giver, taker = world["agents"][offer["from"]], actor
                if giver["resources"][offer["give_resource"]] < offer["give_amount"] \
                        or taker["resources"][offer["want_resource"]] < offer["want_amount"] \
                        or not giver["alive"]:
                    offer["status"] = "failed"
                    fail("TRADE_UNFUNDED", f"Trade {offer['id']} collapsed: one side lacked the resources.")
                else:
                    giver["resources"][offer["give_resource"]] -= offer["give_amount"]
                    taker["resources"][offer["give_resource"]] += offer["give_amount"]
                    taker["resources"][offer["want_resource"]] -= offer["want_amount"]
                    giver["resources"][offer["want_resource"]] += offer["want_amount"]
                    offer["status"] = "accepted"
                    summary = (f"{name} accepted offer {offer['id']}: {offer['give_amount']} "
                               f"{offer['give_resource']} <-> {offer['want_amount']} {offer['want_resource']}.")
                    interactions.append({"from": aid, "to": offer["from"], "type": "accept"})
                    self.say(summary, aid)
        elif kind == "vote":
            world["votes"][action["vote_id"]]["ballots"][aid] = action["option"]
            summary = f"{name} voted '{action['option']}' on {action['vote_id']}."
            self.say(f"{name} cast a ballot on {action['vote_id']}.", aid)
        elif kind == "take":
            victim = world["agents"][target]
            rng = random.Random(f"{self.seed}:{world['turn']}:{aid}:take")
            cons = spec["constraints"]
            grab = min(amt, victim["resources"][res])
            if grab < 1 or not victim["alive"]:
                fail("NOTHING_TO_TAKE", f"{name} tried to take {res} from {_name(spec, target)} but found none.")
            elif rng.random() < cons["take_success_rate"]:
                victim["resources"][res] -= grab
                actor["resources"][res] += grab
                summary = f"{name} secretly took {grab} {res} from {_name(spec, target)}."
                interactions.append({"from": aid, "to": target, "type": "take", "resource": res, "amount": grab})
                world["private_messages"].append({"turn": world["turn"], "from": None, "to": target,
                                                  "from_name": "(unknown)", "message":
                                                  f"Someone took {grab} {res} from you."})
            else:
                fine = min(cons["take_penalty"], actor["resources"][res])
                actor["resources"][res] -= fine
                victim["resources"][res] += fine
                fail("CAUGHT_TAKING", f"{name} was caught trying to take {res} from {_name(spec, target)} "
                                      f"and paid {fine} as a penalty.")
                interactions.append({"from": aid, "to": target, "type": "take_failed", "resource": res, "amount": fine})
                self.say(summary, aid, "exposed")

        after = flatten(world, spec)
        return {"ok": ok, "flags": flags, "summary": summary,
                "changes": diff(before, after), "interactions": interactions}

    def _maybe_complete(self, project: str, aid: str) -> None:
        state, info = self.world["projects"][project], self.spec["projects"][project]
        if not state["done"] and all(state["progress"][r] >= n for r, n in info["needs"].items()):
            state["done"] = True
            for effect in info.get("on_complete", []):
                apply_effect(self.world, self.spec, effect)
            self.say(f"Project '{project}' is complete!", aid, "milestone")


# --------------------------------------------------------------------------- runner
class ScenarioRunner:
    def __init__(self, spec: dict, pool: Optional[ProviderPool] = None, *, seed: int = 7,
                 runs_dir=None, tracker: Optional[UsageTracker] = None, concurrency: int = 4,
                 run_id: Optional[str] = None, progress: Optional[Callable[[dict], None]] = None,
                 turn_limit: Optional[int] = None):
        self.spec = copy.deepcopy(spec)
        if turn_limit is not None:
            self.spec["turn_limit"] = min(turn_limit, self.spec["turn_limit"])
        self.pool = pool or ProviderPool()
        self.seed = seed
        self.rules = ActionRules(self.spec)
        self.tracker = tracker or UsageTracker(max_cost_usd=self.pool.config.get("max_cost_usd"))
        self.concurrency = max(1, concurrency)
        self.run_id = run_id or f"{self.spec['id']}-{uuid.uuid4().hex[:8]}"
        self.log = EventLog(self.run_id, runs_dir)
        self.progress = progress
        self.world = initial_world(self.spec)
        self.resolved = {a["id"]: self.pool.resolve(a) for a in self.spec["agents"]}

    # -- LLM call ------------------------------------------------------------
    def _decide(self, aid: str, perception: dict) -> dict:
        definition = _agent(self.spec, aid)
        info = self.resolved[aid]
        provider = self.pool.get(info["provider"], base_url=info.get("base_url"))
        params = info["params"]
        system = SYSTEM_PROMPT.format(name=definition["name"], role=definition["role"],
                                      title=self.spec["title"])
        request = LLMRequest(
            model=info["model"],
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(perception, ensure_ascii=False)}],
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 2048),
            json_schema=self.rules.json_schema(),
            reasoning_effort=params.get("reasoning_effort"),
            top_p=params.get("top_p"), seed=params.get("seed"),
            metadata={"agent_id": aid, "turn": perception["turn"],
                      "disposition": definition.get("mock_disposition"),
                      "candidates": self.rules.candidates(perception) if info["provider"] == "mock" else None},
        )
        llm = {"provider": info["provider"], "model": info["model"], "tier": info["tier"]}
        flags: List[str] = []
        try:
            result = request_structured(provider, request,
                                        lambda raw: self.rules.parse(raw, perception),
                                        max_repairs=self.pool.config.get("max_repairs", 2))
            action, errors, raw = result.value, result.errors, result.raw
            llm.update(model=result.model, latency_ms=round(result.latency_ms, 1),
                       usage=result.usage.to_dict(), cost_usd=result.cost_usd,
                       retries=result.retries, repairs=result.repairs, calls=result.calls, error=None)
            self.tracker.record(aid, result.model, result.usage, result.cost_usd,
                                attempts=1 + result.retries, latency_ms=result.latency_ms)
        except (LLMAuthError, LLMBadRequestError):
            raise  # misconfiguration (bad key/model/params): fail loudly, don't play on as 'pass'
        except LLMError as exc:  # transient failure that outlived retries
            action, errors, raw = None, [], ""
            flags.append("LLM_ERROR")
            llm.update(error=f"{type(exc).__name__}: {exc}"[:500], usage={}, cost_usd=None,
                       retries=0, repairs=0, calls=0, latency_ms=0)
        if action is None:
            flags.append("FALLBACK_PASS")
            action = {k: None for k in ("target", "resource", "amount", "request_resource",
                                        "request_amount", "offer_id", "vote_id", "option",
                                        "project", "message")}
            action.update(action="pass", reasoning="(no valid action; defaulted to pass)")
        return {"action": action, "validation": {"flags": flags, "errors": errors, "raw": raw[:1500]},
                "llm": llm}

    # -- main loop -----------------------------------------------------------
    def run(self) -> dict:
        spec, world, limit = self.spec, self.world, self.spec["turn_limit"]
        agents_meta = []
        for index, agent in enumerate(spec["agents"]):
            info = self.resolved[agent["id"]]
            agents_meta.append({"id": agent["id"], "name": agent["name"], "role": agent["role"],
                                "persona": agent["persona"], "goals": agent["goals"],
                                "provider": info["provider"], "model": info["model"],
                                "tier": info["tier"], "index": index})
        scenario_info = {k: spec[k] for k in ("id", "title", "kind", "summary", "goal", "briefing",
                                              "turn_limit", "rules", "resources", "projects", "votes")}
        scenario_info["success"] = describe_conditions(spec.get("success"))
        scenario_info["failure"] = describe_conditions(spec.get("failure"))
        scenario_info["scoring"] = spec.get("scoring")
        self.log.emit("run_start", scenario=scenario_info, agents=agents_meta, seed=self.seed,
                      provider=self.pool.config["provider"], state=public_state(world))
        outcome, reason, winner = None, "", None
        try:
            for turn in range(1, limit + 1):
                world["turn"] = turn
                events_now = self._scheduled_events(turn)
                self.log.emit("turn_start", turn=turn, state=public_state(world))
                for event in events_now:
                    before = flatten(world, spec)
                    for effect in event.get("effects", []):
                        apply_effect(world, spec, effect)
                    changes = diff(before, flatten(world, spec))
                    world["public_log"].append({"turn": turn, "agent_id": None, "agent": "WORLD",
                                                "kind": "event", "text": f"{event['title']}: {event['description']}"})
                    self.log.emit("world_event", turn=turn, event_id=event["id"], title=event["title"],
                                  description=event["description"], changes=changes)
                living = [a for a, v in world["agents"].items() if v["alive"]]
                perceptions = {aid: build_perception(world, spec, self.rules, aid, events_now)
                               for aid in living}
                with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                    futures = {aid: executor.submit(self._decide, aid, perceptions[aid]) for aid in living}
                    decisions = {aid: f.result() for aid, f in futures.items()}
                order = list(living)
                random.Random(f"{self.seed}:{turn}").shuffle(order)
                resolver = Resolver(world, spec, self.seed)
                for position, aid in enumerate(order):
                    decision = decisions[aid]
                    effect = resolver.resolve(aid, decision["action"])
                    if effect["flags"]:
                        decision["validation"]["flags"] += effect["flags"]
                    step = self.log.emit(
                        "agent_step", turn=turn, agent_id=aid, order=position,
                        perception=perceptions[aid], reasoning=decision["action"].get("reasoning", ""),
                        action=decision["action"], validation=decision["validation"],
                        effect=effect, llm=decision["llm"])
                    if self.progress:
                        self.progress(step)
                turn_report = self._end_of_turn(turn)
                flat = flatten(world, spec)
                s_met, s_detail = evaluate_group(spec.get("success"), flat, turn, limit)
                f_met, f_detail = evaluate_group(spec.get("failure"), flat, turn, limit)
                self.log.emit("turn_end", turn=turn, state=public_state(world), report=turn_report,
                              conditions={"success": s_detail, "failure": f_detail},
                              scores=scores_for(world, spec), usage=self.tracker.summary())
                if f_met:
                    outcome, reason = "failure", "A failure condition was met."
                elif s_met:
                    outcome, reason = "success", "All success conditions were met."
                elif not any(v["alive"] for v in world["agents"].values()):
                    outcome, reason = "failure", "Every agent was eliminated."
                elif self.tracker.over_budget():
                    outcome, reason = "aborted_budget", \
                        f"Stopped: cost cap ${self.tracker.max_cost_usd} reached."
                if outcome:
                    break
            if outcome is None:
                outcome, reason, winner = self._timeout_outcome()
        except Exception as exc:  # keep the log well-formed for the dashboard
            outcome, reason = "error", f"{type(exc).__name__}: {exc}"
            self.log.emit("run_end", outcome=outcome, reason=reason, winner=None,
                          scores=scores_for(world, spec), state=public_state(world),
                          usage=self.tracker.summary())
            raise
        scores = scores_for(world, spec)
        if outcome in ("success", "failure") and scores and winner is None and spec["kind"] == "competition":
            winner = max(scores, key=scores.get)
        self.log.emit("run_end", outcome=outcome, reason=reason, winner=winner, scores=scores,
                      turns_played=world["turn"], state=public_state(world),
                      usage=self.tracker.summary())
        return {"run_id": self.run_id, "outcome": outcome, "reason": reason, "winner": winner,
                "scores": scores, "turns_played": world["turn"], "usage": self.tracker.summary(),
                "log_path": str(self.log.path)}

    def _timeout_outcome(self):
        spec, world = self.spec, self.world
        mode = spec["on_timeout"]
        scores = scores_for(world, spec)
        if mode == "score" and scores:
            top = max(scores.values())
            winners = [a for a, s in scores.items() if s == top]
            winner = winners[0] if len(winners) == 1 else None
            who = _name(spec, winner) if winner else "a tie between " + ", ".join(_name(spec, w) for w in winners)
            return "decided", f"Turn limit reached; highest score: {who} ({top}).", winner
        if mode == "success":
            return "success", "Turn limit reached with no failure condition met.", None
        return "timeout", "Turn limit reached before the success conditions were met.", None

    def _scheduled_events(self, turn: int) -> List[dict]:
        due = []
        for event in self.spec["events"]:
            every = event.get("repeat_every")
            if event["turn"] == turn or (every and turn > event["turn"] and (turn - event["turn"]) % every == 0):
                due.append(event)
        return due

    def _end_of_turn(self, turn: int) -> dict:
        world, spec = self.world, self.spec
        report: Dict[str, object] = {"votes": [], "upkeep": {}, "deaths": [], "expired_offers": []}
        for vote in spec["votes"]:
            if vote["turn"] + vote.get("window", 1) - 1 == turn and world["votes"][vote["id"]]["result"] is None:
                ballots = world["votes"][vote["id"]]["ballots"]
                tally: Dict[str, int] = {}
                for option in ballots.values():
                    tally[option] = tally.get(option, 0) + 1
                top = max(tally.values()) if tally else 0
                leaders = [o for o, n in tally.items() if n == top] if tally else []
                chosen = leaders[0] if len(leaders) == 1 else vote["default_option"]
                world["votes"][vote["id"]]["result"] = chosen
                before = flatten(world, spec)
                for effect in vote["options"][chosen].get("effects", []):
                    apply_effect(world, spec, effect)
                changes = diff(before, flatten(world, spec))
                note = "default option (no majority)" if len(leaders) != 1 else "plurality"
                text = f"Vote '{vote['id']}' decided: {chosen} ({note}); tally {tally or 'no ballots'}."
                world["public_log"].append({"turn": turn, "agent_id": None, "agent": "WORLD",
                                            "kind": "vote_result", "text": text})
                report["votes"].append({"vote_id": vote["id"], "result": chosen, "tally": tally})
                self.log.emit("world_event", turn=turn, event_id=f"vote:{vote['id']}",
                              title=f"Vote result: {vote['id']}", description=text, changes=changes)
        for offer in world["offers"]:
            if offer["status"] == "open" and offer["expires_turn"] <= turn:
                offer["status"] = "expired"
                report["expired_offers"].append(offer["id"])
        order = [a for a, v in world["agents"].items() if v["alive"]]
        random.Random(f"{self.seed}:{turn}:upkeep").shuffle(order)
        for aid in order:
            agent = world["agents"][aid]
            shortfall = False
            for item in spec["upkeep"]:
                res, need, source = item["resource"], item["amount"], item.get("source", "self_then_pool")
                if source in ("self", "self_then_pool"):
                    use = min(need, agent["resources"][res])
                    agent["resources"][res] -= use
                    need -= use
                if need and source in ("pool", "self_then_pool"):
                    use = min(need, world["pool"][res])
                    world["pool"][res] -= use
                    need -= use
                if need:
                    shortfall = True
                    agent["health"] = max(0, agent["health"] + item.get("shortfall_health", 0))
                    report["upkeep"].setdefault(aid, []).append(f"short {need} {res}")
            if not shortfall:
                agent["health"] = min(spec["health"]["initial"],
                                      agent["health"] + spec["health"].get("recovery_per_turn", 0))
            if agent["health"] <= 0:
                kill(world, aid)
                report["deaths"].append(aid)
                world["public_log"].append({"turn": turn, "agent_id": aid, "agent": _name(spec, aid),
                                            "kind": "death", "text": f"{_name(spec, aid)} has died."})
        return report
