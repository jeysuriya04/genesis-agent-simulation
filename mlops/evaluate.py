#!/usr/bin/env python3
"""Evaluate stage: derive metrics from events.jsonl and gate against thresholds/baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis.events import read_events  # noqa: E402
from mlops.io_util import read_json, write_json  # noqa: E402
from mlops.paths import ARTIFACTS_DIR, BASELINES_DIR, THRESHOLDS_DIR, ensure_layout  # noqa: E402


def extract_metrics(events: list[dict]) -> dict[str, Any]:
    if not events:
        raise ValueError("no events to evaluate")
    start = next((e for e in events if e.get("type") == "run_start"), events[0])
    end = next((e for e in reversed(events) if e.get("type") == "run_end"), None)
    if end is None:
        raise ValueError("run_end event missing; experiment incomplete")

    steps = [e for e in events if e.get("type") == "agent_step"]
    flagged = 0
    errored = 0
    for step in steps:
        validation = step.get("validation") or {}
        if validation.get("flags"):
            flagged += 1
        llm = step.get("llm") or {}
        if llm.get("error") or validation.get("errors"):
            errored += 1

    usage = end.get("usage") or {}
    turns_played = end.get("turns_played")
    if turns_played is None:
        turn_ends = [e.get("turn", 0) for e in events if e.get("type") == "turn_end"]
        turns_played = max(turn_ends) if turn_ends else 0

    turn_limit = (start.get("scenario") or {}).get("turn_limit")
    n_steps = len(steps) or 1
    return {
        "run_id": start.get("run_id") or end.get("run_id"),
        "scenario_id": (start.get("scenario") or {}).get("id"),
        "provider": start.get("provider"),
        "outcome": end.get("outcome"),
        "reason": end.get("reason"),
        "winner": end.get("winner"),
        "scores": end.get("scores") or {},
        "turns_played": turns_played,
        "turn_limit": turn_limit,
        "turn_completion": (turns_played / turn_limit) if turn_limit else None,
        "calls": usage.get("calls", 0),
        "retries": usage.get("retries", 0),
        "cost_usd": float(usage.get("cost_usd") or 0.0),
        "total_tokens": usage.get("total_tokens", 0),
        "steps": len(steps),
        "flag_rate": flagged / n_steps,
        "error_rate": errored / n_steps,
        "flagged_steps": flagged,
        "errored_steps": errored,
    }


def _check_thresholds(metrics: dict, thresholds: dict) -> list[str]:
    problems: list[str] = []
    outcome = metrics.get("outcome")
    forbidden = set(thresholds.get("forbidden_outcomes") or ["error"])
    if outcome in forbidden:
        problems.append(f"outcome {outcome!r} is forbidden")

    allowed = thresholds.get("allowed_outcomes")
    if allowed is not None and outcome not in allowed:
        problems.append(f"outcome {outcome!r} not in allowed_outcomes {allowed}")

    def _max(key: str, metric_key: str | None = None) -> None:
        limit = thresholds.get(key)
        if limit is None:
            return
        value = metrics[metric_key or key.replace("max_", "")]
        if value > limit:
            problems.append(f"{metric_key or key}={value} exceeds {key}={limit}")

    def _min(key: str, metric_key: str | None = None) -> None:
        limit = thresholds.get(key)
        if limit is None:
            return
        value = metrics[metric_key or key.replace("min_", "")]
        if value < limit:
            problems.append(f"{metric_key or key}={value} below {key}={limit}")

    _max("max_retries", "retries")
    _max("max_cost_usd", "cost_usd")
    _max("max_flag_rate", "flag_rate")
    _max("max_error_rate", "error_rate")
    _min("min_turns_played", "turns_played")
    _min("min_calls", "calls")
    return problems


def _check_baseline(metrics: dict, baseline: dict) -> list[str]:
    problems: list[str] = []
    expected = baseline.get("metrics") or {}
    tolerance = baseline.get("tolerance") or {}

    if "outcome" in expected and metrics.get("outcome") != expected["outcome"]:
        problems.append(
            f"baseline outcome expected {expected['outcome']!r}, got {metrics.get('outcome')!r}"
        )

    for key, target in expected.items():
        if key == "outcome":
            continue
        if key not in metrics:
            continue
        actual = metrics[key]
        if isinstance(target, (int, float)) and isinstance(actual, (int, float)):
            band = tolerance.get(key, 0)
            if abs(actual - target) > band:
                problems.append(f"baseline {key}: expected {target}±{band}, got {actual}")
        elif actual != target:
            problems.append(f"baseline {key}: expected {target!r}, got {actual!r}")
    return problems


def evaluate_run(
    log_path: Path,
    *,
    scenario_id: Optional[str] = None,
    thresholds_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
    provider: Optional[str] = None,
) -> dict:
    events = read_events(log_path)
    metrics = extract_metrics(events)
    scenario_id = scenario_id or metrics.get("scenario_id")
    provider = provider or metrics.get("provider") or "mock"

    if thresholds_path is None and scenario_id:
        candidate = THRESHOLDS_DIR / f"{scenario_id}.json"
        thresholds_path = candidate if candidate.exists() else None
    if baseline_path is None and scenario_id:
        candidate = BASELINES_DIR / f"{scenario_id}.{provider}.json"
        baseline_path = candidate if candidate.exists() else None

    thresholds = read_json(thresholds_path) if thresholds_path and thresholds_path.exists() else {}
    baseline = read_json(baseline_path) if baseline_path and baseline_path.exists() else None

    reasons = _check_thresholds(metrics, thresholds)
    if baseline:
        reasons.extend(_check_baseline(metrics, baseline))

    passed = not reasons
    return {
        "passed": passed,
        "reasons": reasons,
        "metrics": metrics,
        "thresholds_path": str(thresholds_path) if thresholds_path else None,
        "baseline_path": str(baseline_path) if baseline_path else None,
        "log_path": str(log_path),
    }


def evaluate_artifact(exp_id: str) -> dict:
    ensure_layout()
    artifact_dir = ARTIFACTS_DIR / exp_id
    manifest = read_json(artifact_dir / "manifest.json")
    decision = evaluate_run(
        Path(manifest["log_path"]),
        scenario_id=manifest.get("scenario_id"),
        provider=manifest.get("provider"),
    )
    write_json(artifact_dir / "metrics.json", decision["metrics"])
    write_json(
        artifact_dir / "decision.json",
        {
            "passed": decision["passed"],
            "reasons": decision["reasons"],
            "thresholds_path": decision["thresholds_path"],
            "baseline_path": decision["baseline_path"],
            "exp_id": exp_id,
            "run_id": manifest.get("run_id"),
        },
    )
    status = "PASSED" if decision["passed"] else "FAILED"
    print(f"EVALUATE {status}: exp_id={exp_id} outcome={decision['metrics'].get('outcome')}")
    for reason in decision["reasons"]:
        print(f"  - {reason}")
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Genesis run or MLOps artifact.")
    parser.add_argument("--exp-id", help="Evaluate mlops/artifacts/<exp_id>")
    parser.add_argument("--log", type=Path, help="Path to events.jsonl")
    parser.add_argument("--scenario")
    parser.add_argument("--provider")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)

    if args.exp_id:
        decision = evaluate_artifact(args.exp_id)
        return 0 if decision["passed"] else 1
    if not args.log:
        parser.error("provide --exp-id or --log")
    decision = evaluate_run(
        args.log,
        scenario_id=args.scenario,
        thresholds_path=args.thresholds,
        baseline_path=args.baseline,
        provider=args.provider,
    )
    print(json_dumps(decision))
    return 0 if decision["passed"] else 1


def json_dumps(data: dict) -> str:
    import json
    return json.dumps(data, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
