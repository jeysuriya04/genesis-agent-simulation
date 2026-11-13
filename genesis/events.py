"""Structured JSONL event log: the single source of truth for monitoring and replay.

One file per run at ``runs/<run_id>/events.jsonl``. Every line is a JSON object with
``seq`` (1-based, monotonic), ``ts`` (UTC ISO), ``run_id`` and ``type``:

* ``run_start``  - scenario summary, agents (provider/model), seed, initial state
* ``turn_start`` - turn number and world state before anyone acts
* ``world_event``- scheduled event / vote result with its state changes
* ``agent_step`` - one agent's turn: perception, reasoning, action, effect, LLM stats
* ``turn_end``   - state after resolution, success/failure condition status, usage
* ``run_end``    - outcome, reason, winner/scores, final state, usage summary

``reasoning`` is the model's self-reported rationale, not verified internal reasoning.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

EVENT_TYPES = ("run_start", "turn_start", "world_event", "agent_step", "turn_end", "run_end")
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLog:
    """Append-only, thread-safe, flushed per event so a live dashboard can tail it."""

    def __init__(self, run_id: str, runs_dir=None):
        self.run_id = run_id
        self.dir = Path(runs_dir or DEFAULT_RUNS_DIR) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._lock = threading.Lock()
        self.seq = 0

    def emit(self, type_: str, **payload) -> dict:
        if type_ not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {type_}")
        with self._lock:
            self.seq += 1
            event = {"seq": self.seq, "ts": utc_now(), "run_id": self.run_id, "type": type_, **payload}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        return event


def read_events(path, after: int = 0) -> List[dict]:
    """Read events with ``seq > after``; ignores a half-written trailing line."""
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial write from a live run
            if event.get("seq", 0) > after:
                events.append(event)
    return events


def iter_runs(runs_dir=None) -> Iterator[Path]:
    root = Path(runs_dir or DEFAULT_RUNS_DIR)
    if root.exists():
        for child in sorted(root.iterdir()):
            if (child / "events.jsonl").exists():
                yield child


def summarize_run(run_dir) -> Optional[dict]:
    """Small listing record for the dashboard's run picker."""
    run_dir = Path(run_dir)
    events = read_events(run_dir / "events.jsonl")
    if not events or events[0]["type"] != "run_start":
        return None
    start = events[0]
    end = next((e for e in reversed(events) if e["type"] == "run_end"), None)
    turns = [e["turn"] for e in events if e["type"] == "turn_end"]
    return {
        "run_id": start["run_id"],
        "scenario_id": start["scenario"]["id"],
        "title": start["scenario"]["title"],
        "kind": start["scenario"].get("kind"),
        "provider": start.get("provider"),
        "started_at": start["ts"],
        "turn_limit": start["scenario"]["turn_limit"],
        "turns_done": max(turns) if turns else 0,
        "status": end["outcome"] if end else "running",
        "events": len(events),
        "cost_usd": (end or {}).get("usage", {}).get("cost_usd"),
    }
