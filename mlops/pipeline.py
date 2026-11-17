#!/usr/bin/env python3
"""Orchestrate validate → test → experiment → evaluate → register [→ promote] [→ release]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlops.evaluate import evaluate_artifact  # noqa: E402
from mlops.paths import ensure_layout  # noqa: E402
from mlops.promote import promote_entry  # noqa: E402
from mlops.register import register_artifact  # noqa: E402
from mlops.release import release_entry  # noqa: E402
from mlops.run_experiment import run_experiment  # noqa: E402
from mlops.run_tests import run_tests  # noqa: E402
from mlops.validate_configs import main as validate_main  # noqa: E402


def run_pipeline(args: argparse.Namespace) -> int:
    ensure_layout()
    skip = set(args.skip or [])

    if "validate" not in skip:
        code = validate_main([])
        if code != 0:
            return code

    if "test" not in skip:
        code = run_tests()
        if code != 0:
            return code

    if "experiment" not in skip:
        manifest = run_experiment(
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
        exp_id = manifest["exp_id"]
    else:
        if not args.exp_id:
            print("error: --exp-id required when skipping experiment", file=sys.stderr)
            return 1
        exp_id = args.exp_id

    if "evaluate" not in skip:
        decision = evaluate_artifact(exp_id)
        if not decision["passed"] and not args.allow_fail:
            return 1

    if "register" not in skip:
        try:
            entry = register_artifact(exp_id, require_pass=not args.allow_fail, stage="candidate")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        entry = {"exp_id": exp_id}

    if args.promote:
        try:
            # walk stages in order up to the requested target
            from mlops.register import load_index
            from mlops.promote import find_entry, ORDER
            current = find_entry(load_index(), exp_id=exp_id)
            target = args.promote
            while ORDER[current["stage"]] < ORDER[target]:
                nxt = ["candidate", "staging", "production"][ORDER[current["stage"]] + 1]
                current = promote_entry(
                    target_stage=nxt,
                    exp_id=exp_id,
                    require_pass=not args.allow_fail,
                )
            entry = current
        except (LookupError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.release:
        try:
            release_entry(
                exp_id=exp_id,
                dry_run=args.dry_run_release,
                skip_gh=args.skip_gh,
            )
        except (LookupError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(f"PIPELINE OK: exp_id={exp_id} tag={entry.get('tag', 'n/a')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genesis MLOps pipeline: validate → test → experiment → evaluate → register.",
    )
    parser.add_argument("--scenario", default="ram-setu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", choices=["xai", "groq", "openai", "mock"], default="mock")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--no-mock", action="store_false", dest="mock")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--turns", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--model-main")
    parser.add_argument("--model-light")
    parser.add_argument("--exp-id", help="Reuse/force experiment id (required with --skip experiment)")
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["validate", "test", "experiment", "evaluate", "register"],
        help="Skip named stages",
    )
    parser.add_argument("--promote", choices=["staging", "production"],
                        help="After register, promote up to this stage")
    parser.add_argument("--release", action="store_true",
                        help="Create git tag + GitHub release (production only)")
    parser.add_argument("--dry-run-release", action="store_true")
    parser.add_argument("--skip-gh", action="store_true")
    parser.add_argument("--allow-fail", action="store_true",
                        help="Continue register/promote even if evaluate failed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
