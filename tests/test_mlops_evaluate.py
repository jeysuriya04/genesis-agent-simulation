"""Unit tests for MLOps evaluate metrics and gating."""
from __future__ import annotations

import json
from pathlib import Path

from mlops.evaluate import evaluate_run, extract_metrics
from mlops.paths import BASELINES_DIR, THRESHOLDS_DIR


def _write_events(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return path


def _fixture_events(outcome="timeout", flags=None, llm_error=None, retries=0):
    flags = flags or []
    return [
        {
            "seq": 1,
            "ts": "2026-01-01T00:00:00.000+00:00",
            "run_id": "ram-setu-test",
            "type": "run_start",
            "provider": "mock",
            "scenario": {"id": "ram-setu", "title": "Ram Setu", "kind": "cooperation", "turn_limit": 10},
        },
        {
            "seq": 2,
            "ts": "2026-01-01T00:00:01.000+00:00",
            "run_id": "ram-setu-test",
            "type": "agent_step",
            "turn": 1,
            "agent_id": "nala",
            "validation": {"flags": flags, "errors": []},
            "llm": {"error": llm_error},
        },
        {
            "seq": 3,
            "ts": "2026-01-01T00:00:02.000+00:00",
            "run_id": "ram-setu-test",
            "type": "agent_step",
            "turn": 1,
            "agent_id": "hanuman",
            "validation": {"flags": [], "errors": []},
            "llm": {},
        },
        {
            "seq": 4,
            "ts": "2026-01-01T00:00:03.000+00:00",
            "run_id": "ram-setu-test",
            "type": "turn_end",
            "turn": 10,
        },
        {
            "seq": 5,
            "ts": "2026-01-01T00:00:04.000+00:00",
            "run_id": "ram-setu-test",
            "type": "run_end",
            "outcome": outcome,
            "reason": "fixture",
            "winner": None,
            "scores": {},
            "turns_played": 10,
            "usage": {
                "calls": 50,
                "retries": retries,
                "cost_usd": 0.0,
                "total_tokens": 100,
            },
        },
    ]


def test_extract_metrics_rates(tmp_path):
    log = _write_events(tmp_path / "events.jsonl", _fixture_events(flags=["X"], llm_error="boom"))
    from genesis.events import read_events

    metrics = extract_metrics(read_events(log))
    assert metrics["outcome"] == "timeout"
    assert metrics["turns_played"] == 10
    assert metrics["calls"] == 50
    assert metrics["steps"] == 2
    assert metrics["flag_rate"] == 0.5
    assert metrics["error_rate"] == 0.5


def test_thresholds_forbid_error(tmp_path):
    log = _write_events(tmp_path / "events.jsonl", _fixture_events(outcome="error"))
    decision = evaluate_run(
        log,
        scenario_id="ram-setu",
        thresholds_path=THRESHOLDS_DIR / "ram-setu.json",
        baseline_path=None,
        provider="mock",
    )
    assert decision["passed"] is False
    assert any("forbidden" in r for r in decision["reasons"])


def test_ram_setu_mock_baseline_pass(tmp_path):
    log = _write_events(tmp_path / "events.jsonl", _fixture_events())
    decision = evaluate_run(
        log,
        scenario_id="ram-setu",
        thresholds_path=THRESHOLDS_DIR / "ram-setu.json",
        baseline_path=BASELINES_DIR / "ram-setu.mock.json",
        provider="mock",
    )
    assert decision["passed"] is True
    assert decision["reasons"] == []


def test_baseline_call_regression(tmp_path):
    events = _fixture_events()
    events[-1]["usage"]["calls"] = 40
    log = _write_events(tmp_path / "events.jsonl", events)
    decision = evaluate_run(
        log,
        scenario_id="ram-setu",
        thresholds_path=THRESHOLDS_DIR / "ram-setu.json",
        baseline_path=BASELINES_DIR / "ram-setu.mock.json",
        provider="mock",
    )
    assert decision["passed"] is False
    assert any("calls" in r for r in decision["reasons"])
