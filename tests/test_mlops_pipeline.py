"""Smoke tests for MLOps register/promote and evaluate-on-artifact path."""
from __future__ import annotations

import json
from pathlib import Path

import mlops.paths as paths
from mlops.evaluate import evaluate_artifact
from mlops.io_util import write_json
from mlops.promote import promote_entry
from mlops.register import load_index, register_artifact, save_index


def test_register_promote_flow(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    registry = tmp_path / "registry"
    artifacts.mkdir()
    registry.mkdir()
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(paths, "REGISTRY_DIR", registry)
    monkeypatch.setattr(paths, "REGISTRY_INDEX", registry / "index.json")

    # re-bind modules that imported paths at load time
    import mlops.evaluate as evaluate_mod
    import mlops.register as register_mod
    import mlops.promote as promote_mod

    monkeypatch.setattr(evaluate_mod, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(register_mod, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(register_mod, "REGISTRY_INDEX", registry / "index.json")
    monkeypatch.setattr(promote_mod, "ARTIFACTS_DIR", artifacts)

    save_index({"entries": []})

    exp_id = "exp-smoke"
    run_dir = tmp_path / "runs" / "ram-setu-smoke"
    run_dir.mkdir(parents=True)
    events = [
        {
            "seq": 1,
            "ts": "2026-01-01T00:00:00.000+00:00",
            "run_id": "ram-setu-smoke",
            "type": "run_start",
            "provider": "mock",
            "scenario": {"id": "ram-setu", "title": "Ram Setu", "kind": "cooperation", "turn_limit": 10},
        },
        {
            "seq": 2,
            "ts": "2026-01-01T00:00:01.000+00:00",
            "run_id": "ram-setu-smoke",
            "type": "agent_step",
            "turn": 1,
            "agent_id": "nala",
            "validation": {"flags": [], "errors": []},
            "llm": {},
        },
        {
            "seq": 3,
            "ts": "2026-01-01T00:00:02.000+00:00",
            "run_id": "ram-setu-smoke",
            "type": "run_end",
            "outcome": "timeout",
            "reason": "fixture",
            "winner": None,
            "scores": {},
            "turns_played": 10,
            "usage": {"calls": 50, "retries": 0, "cost_usd": 0.0, "total_tokens": 10},
        },
    ]
    log_path = run_dir / "events.jsonl"
    log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    artifact_dir = artifacts / exp_id
    write_json(
        artifact_dir / "manifest.json",
        {
            "exp_id": exp_id,
            "scenario_id": "ram-setu",
            "provider": "mock",
            "seed": 0,
            "run_id": "ram-setu-smoke",
            "log_path": str(log_path),
            "outcome": "timeout",
            "hashes": {},
        },
    )

    decision = evaluate_artifact(exp_id)
    assert decision["passed"] is True

    entry = register_artifact(exp_id)
    assert entry["stage"] == "candidate"
    assert entry["tag"] == "genesis-ram-setu-v1"

    staging = promote_entry(target_stage="staging", exp_id=exp_id)
    assert staging["stage"] == "staging"
    production = promote_entry(target_stage="production", exp_id=exp_id)
    assert production["stage"] == "production"

    index = load_index()
    assert len(index["entries"]) == 1
    assert index["entries"][0]["stage"] == "production"
