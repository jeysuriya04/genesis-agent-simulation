#!/usr/bin/env python3
"""Register stage: record a passed (or forced) experiment in the file registry."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlops.io_util import read_json, write_json  # noqa: E402
from mlops.paths import ARTIFACTS_DIR, REGISTRY_INDEX, ensure_layout  # noqa: E402

STAGES = ("candidate", "staging", "production")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_index() -> dict[str, Any]:
    ensure_layout()
    if REGISTRY_INDEX.exists():
        return read_json(REGISTRY_INDEX)
    return {"entries": []}


def save_index(index: dict[str, Any]) -> None:
    ensure_layout()
    write_json(REGISTRY_INDEX, index)


def next_version(index: dict[str, Any], scenario_id: str) -> int:
    versions = [
        int(entry.get("version", 0))
        for entry in index.get("entries", [])
        if entry.get("scenario_id") == scenario_id
    ]
    return (max(versions) if versions else 0) + 1


def register_artifact(exp_id: str, *, require_pass: bool = True, stage: str = "candidate") -> dict:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")

    artifact_dir = ARTIFACTS_DIR / exp_id
    manifest = read_json(artifact_dir / "manifest.json")
    decision_path = artifact_dir / "decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(f"missing {decision_path}; run evaluate first")
    decision = read_json(decision_path)
    if require_pass and not decision.get("passed"):
        raise RuntimeError(
            f"refusing to register failed evaluation for {exp_id}: {decision.get('reasons')}"
        )

    index = load_index()
    scenario_id = manifest["scenario_id"]
    version = next_version(index, scenario_id)
    entry = {
        "version": version,
        "exp_id": exp_id,
        "scenario_id": scenario_id,
        "stage": stage,
        "registered_at": utc_now(),
        "promoted_at": utc_now() if stage != "candidate" else None,
        "run_id": manifest.get("run_id"),
        "provider": manifest.get("provider"),
        "seed": manifest.get("seed"),
        "outcome": manifest.get("outcome"),
        "passed": bool(decision.get("passed")),
        "hashes": manifest.get("hashes") or {},
        "artifact_dir": str(artifact_dir),
        "metrics_path": str(artifact_dir / "metrics.json"),
        "decision_path": str(decision_path),
        "log_path": manifest.get("log_path"),
        "tag": f"genesis-{scenario_id}-v{version}",
    }
    index.setdefault("entries", []).append(entry)
    save_index(index)
    write_json(artifact_dir / "registry_entry.json", entry)
    print(f"REGISTERED: {entry['tag']} stage={stage} exp_id={exp_id}")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register an evaluated Genesis experiment.")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--stage", choices=STAGES, default="candidate")
    parser.add_argument("--allow-fail", action="store_true", help="Register even if evaluate failed")
    args = parser.parse_args(argv)
    try:
        register_artifact(args.exp_id, require_pass=not args.allow_fail, stage=args.stage)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
