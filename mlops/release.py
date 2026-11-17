#!/usr/bin/env python3
"""Release stage: create a git tag + GitHub Release for a production registry entry."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlops.io_util import read_json  # noqa: E402
from mlops.paths import ARTIFACTS_DIR, ROOT as PROJECT_ROOT  # noqa: E402
from mlops.promote import find_entry  # noqa: E402
from mlops.register import load_index  # noqa: E402


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"RELEASE: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check, text=True, capture_output=True)


def build_notes(entry: dict) -> str:
    metrics_path = Path(entry.get("metrics_path") or ARTIFACTS_DIR / entry["exp_id"] / "metrics.json")
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    lines = [
        f"# Genesis release `{entry['tag']}`",
        "",
        f"- Scenario: `{entry.get('scenario_id')}`",
        f"- Stage: `{entry.get('stage')}`",
        f"- Experiment: `{entry.get('exp_id')}`",
        f"- Run: `{entry.get('run_id')}`",
        f"- Provider: `{entry.get('provider')}`",
        f"- Seed: `{entry.get('seed')}`",
        f"- Outcome: `{entry.get('outcome')}`",
        f"- Eval passed: `{entry.get('passed')}`",
        "",
        "## Metrics",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def release_entry(
    *,
    exp_id: Optional[str] = None,
    tag: Optional[str] = None,
    scenario_id: Optional[str] = None,
    version: Optional[int] = None,
    dry_run: bool = False,
    skip_gh: bool = False,
) -> dict:
    index = load_index()
    entry = find_entry(
        index, exp_id=exp_id, tag=tag, scenario_id=scenario_id, version=version
    )
    if entry.get("stage") != "production":
        raise RuntimeError(
            f"{entry.get('tag')} is at stage {entry.get('stage')!r}; promote to production first"
        )
    release_tag = entry["tag"]
    notes = build_notes(entry)
    notes_path = ARTIFACTS_DIR / entry["exp_id"] / "release_notes.md"
    notes_path.write_text(notes, encoding="utf-8")

    if dry_run:
        print(f"RELEASE DRY-RUN: would tag and publish {release_tag}")
        print(notes)
        return entry

    # Create annotated tag if missing
    existing = _run(["git", "tag", "-l", release_tag], check=False)
    if release_tag not in (existing.stdout or "").splitlines():
        _run(["git", "tag", "-a", release_tag, "-m", f"Genesis release {release_tag}"])
        push = _run(["git", "push", "origin", release_tag], check=False)
        if push.returncode != 0:
            print(push.stderr or push.stdout, file=sys.stderr)
            raise RuntimeError("failed to push tag; create/push manually then re-run release")
    else:
        print(f"RELEASE: tag {release_tag} already exists")

    if not skip_gh:
        gh = _run(
            [
                "gh", "release", "create", release_tag,
                "--title", release_tag,
                "--notes-file", str(notes_path),
            ],
            check=False,
        )
        if gh.returncode != 0:
            # idempotent-ish: if release exists, edit notes
            if "already exists" in (gh.stderr or "").lower() or "already exists" in (gh.stdout or "").lower():
                _run(["gh", "release", "edit", release_tag, "--notes-file", str(notes_path)])
            else:
                print(gh.stderr or gh.stdout, file=sys.stderr)
                raise RuntimeError("gh release create failed")
    print(f"RELEASED: {release_tag}")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release a production Genesis registry entry.")
    parser.add_argument("--exp-id")
    parser.add_argument("--tag")
    parser.add_argument("--scenario")
    parser.add_argument("--version", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gh", action="store_true", help="Only create/push the git tag")
    args = parser.parse_args(argv)
    if not any([args.exp_id, args.tag, args.scenario]):
        parser.error("provide --exp-id, --tag, or --scenario")
    try:
        release_entry(
            exp_id=args.exp_id,
            tag=args.tag,
            scenario_id=args.scenario,
            version=args.version,
            dry_run=args.dry_run,
            skip_gh=args.skip_gh,
        )
    except (LookupError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
