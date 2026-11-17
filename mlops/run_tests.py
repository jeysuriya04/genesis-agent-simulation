#!/usr/bin/env python3
"""Test stage: run the project pytest suite."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mlops.paths import ROOT


def run_tests(root: Path | None = None, extra_args: list[str] | None = None) -> int:
    root = root or ROOT
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"TEST: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=root)
    if completed.returncode == 0:
        print("TEST PASSED")
    else:
        print("TEST FAILED")
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Genesis unit tests.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("pytest_args", nargs="*", help="Extra args forwarded to pytest")
    args = parser.parse_args(argv)
    return run_tests(args.root, args.pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
