#!/usr/bin/env python3
"""Promote stage: move a registry entry candidate → staging → production."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Optional

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlops.io_util import write_json  # noqa: E402
from mlops.paths import ARTIFACTS_DIR  # noqa: E402
from mlops.register import STAGES, load_index, save_index  # noqa: E402

ORDER = {name: i for i, name in enumerate(STAGES)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def find_entry(index: dict, *, exp_id: Optional[str] = None, tag: Optional[str] = None,
               scenario_id: Optional[str] = None, version: Optional[int] = None) -> dict:
    entries = index.get("entries") or []
    matches = []
    for entry in entries:
        if exp_id and entry.get("exp_id") != exp_id:
            continue
        if tag and entry.get("tag") != tag:
            continue
        if scenario_id and entry.get("scenario_id") != scenario_id:
            continue
        if version is not None and int(entry.get("version", -1)) != int(version):
            continue
        if exp_id or tag or (scenario_id and version is not None):
            matches.append(entry)
        elif scenario_id and entry.get("scenario_id") == scenario_id:
            matches.append(entry)
    if not matches:
        raise LookupError("no matching registry entry")
    # prefer highest version when selecting by scenario alone
    return sorted(matches, key=lambda e: int(e.get("version", 0)))[-1]


def promote_entry(
    *,
    target_stage: str,
    exp_id: Optional[str] = None,
    tag: Optional[str] = None,
    scenario_id: Optional[str] = None,
    version: Optional[int] = None,
    require_pass: bool = True,
) -> dict:
    if target_stage not in STAGES:
        raise ValueError(f"target stage must be one of {STAGES}")

    index = load_index()
    entry = find_entry(
        index, exp_id=exp_id, tag=tag, scenario_id=scenario_id, version=version
    )
    if require_pass and not entry.get("passed"):
        raise RuntimeError(f"cannot promote failed entry {entry.get('tag')}")

    current = entry.get("stage", "candidate")
    if ORDER[target_stage] < ORDER[current]:
        raise RuntimeError(f"cannot demote {current} → {target_stage}")
    if ORDER[target_stage] == ORDER[current]:
        print(f"PROMOTE: {entry['tag']} already at {current}")
        return entry
    if ORDER[target_stage] - ORDER[current] > 1:
        raise RuntimeError(
            f"promote one stage at a time ({current} → {STAGES[ORDER[current] + 1]}), "
            f"not jump to {target_stage}"
        )

    entry["stage"] = target_stage
    entry["promoted_at"] = utc_now()
    save_index(index)

    artifact_dir = ARTIFACTS_DIR / entry["exp_id"]
    write_json(artifact_dir / "registry_entry.json", entry)
    print(f"PROMOTED: {entry['tag']} {current} → {target_stage}")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a Genesis registry entry.")
    parser.add_argument("--to", required=True, choices=["staging", "production"])
    parser.add_argument("--exp-id")
    parser.add_argument("--tag")
    parser.add_argument("--scenario")
    parser.add_argument("--version", type=int)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args(argv)
    if not any([args.exp_id, args.tag, args.scenario]):
        parser.error("provide --exp-id, --tag, or --scenario")
    try:
        promote_entry(
            target_stage=args.to,
            exp_id=args.exp_id,
            tag=args.tag,
            scenario_id=args.scenario,
            version=args.version,
            require_pass=not args.allow_fail,
        )
    except (LookupError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
