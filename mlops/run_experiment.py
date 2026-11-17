#!/usr/bin/env python3
"""Experiment stage: run a scenario and write an artifact manifest."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis.engine import ScenarioRunner  # noqa: E402
from genesis.llm import ProviderPool, UsageTracker, load_env, merge_llm_config  # noqa: E402
from genesis.scenario_spec import get_scenario  # noqa: E402
from mlops.io_util import file_sha256, write_json  # noqa: E402
from mlops.paths import ARTIFACTS_DIR, DEFAULT_RUNS_DIR, ROOT as PROJECT_ROOT, ensure_layout  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_experiment(
    scenario: str,
    *,
    seed: int = 0,
    provider: str = "mock",
    mock: bool = False,
    runs_dir: Path | None = None,
    turns: int | None = None,
    concurrency: int | None = None,
    max_cost: float | None = None,
    model_main: str | None = None,
    model_light: str | None = None,
    exp_id: str | None = None,
) -> dict:
    """Execute one scenario run and persist ``mlops/artifacts/<exp_id>/manifest.json``."""
    ensure_layout()
    load_env()
    if mock:
        provider = "mock"

    base = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
    override: dict = {"provider": provider}
    tiers = {}
    if model_main:
        tiers["main"] = {"model": model_main}
    if model_light:
        tiers["light"] = {"model": model_light}
    if tiers:
        override["tiers"] = tiers
    if max_cost is not None:
        override["max_cost_usd"] = max_cost
    llm = merge_llm_config(base.get("llm"), override)

    spec = get_scenario(scenario)
    runs_dir = Path(runs_dir or DEFAULT_RUNS_DIR)
    pool = ProviderPool(llm, pricing=base.get("pricing"), mock_seed=seed)
    tracker = UsageTracker(max_cost_usd=llm.get("max_cost_usd"))
    runner = ScenarioRunner(
        spec,
        pool,
        seed=seed,
        runs_dir=runs_dir,
        tracker=tracker,
        concurrency=concurrency or base.get("max_concurrency", 4),
        turn_limit=turns,
    )
    print(
        f"EXPERIMENT: scenario={spec['id']} provider={llm['provider']} "
        f"seed={seed} run={runner.run_id}"
    )
    result = runner.run()

    exp_id = exp_id or f"{spec['id']}-{utc_now().replace(':', '').replace('+', 'Z')}-{uuid.uuid4().hex[:8]}"
    artifact_dir = ARTIFACTS_DIR / exp_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scenario_path = PROJECT_ROOT / "scenarios" / f"{spec['id']}.json"
    if not scenario_path.exists():
        # custom path scenarios still hash the validated spec
        scenario_hash = file_sha256(Path(scenario)) if Path(scenario).exists() else ""
    else:
        scenario_hash = file_sha256(scenario_path)

    manifest = {
        "exp_id": exp_id,
        "created_at": utc_now(),
        "scenario_id": spec["id"],
        "scenario_kind": spec.get("kind"),
        "scenario_title": spec.get("title"),
        "seed": seed,
        "provider": llm["provider"],
        "run_id": result["run_id"],
        "log_path": result.get("log_path") or str(runs_dir / result["run_id"] / "events.jsonl"),
        "runs_dir": str(runs_dir),
        "outcome": result.get("outcome"),
        "reason": result.get("reason"),
        "turns_played": result.get("turns_played"),
        "scores": result.get("scores") or {},
        "usage": result.get("usage") or {},
        "hashes": {
            "config": file_sha256(PROJECT_ROOT / "config.json"),
            "scenario": scenario_hash,
            "scenarios_schedule": file_sha256(PROJECT_ROOT / "scenarios.json"),
        },
        "llm": {
            "provider": llm.get("provider"),
            "tiers": llm.get("tiers"),
            "max_cost_usd": llm.get("max_cost_usd"),
        },
    }
    write_json(artifact_dir / "manifest.json", manifest)
    write_json(artifact_dir / "result.json", result)
    print(f"EXPERIMENT DONE: exp_id={exp_id} outcome={result.get('outcome')} "
          f"events={manifest['log_path']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Genesis scenario experiment for MLOps.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", choices=["xai", "groq", "openai", "mock"], default="mock")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--turns", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--model-main")
    parser.add_argument("--model-light")
    parser.add_argument("--exp-id")
    args = parser.parse_args(argv)
    try:
        run_experiment(
            args.scenario,
            seed=args.seed,
            provider=args.provider,
            mock=args.mock,
            runs_dir=args.runs_dir,
            turns=args.turns,
            concurrency=args.concurrency,
            max_cost=args.max_cost,
            model_main=args.model_main,
            model_light=args.model_light,
            exp_id=args.exp_id,
        )
    except Exception as exc:  # noqa: BLE001
        from genesis.llm import LLMAuthError
        if isinstance(exc, LLMAuthError):
            print(f"error: {exc}\n(or run with --mock)", file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
