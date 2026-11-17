"""Shared filesystem roots for the Genesis MLOps loop."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLOPS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MLOPS_DIR / "artifacts"
REGISTRY_DIR = MLOPS_DIR / "registry"
REGISTRY_INDEX = REGISTRY_DIR / "index.json"
THRESHOLDS_DIR = MLOPS_DIR / "thresholds"
BASELINES_DIR = MLOPS_DIR / "baselines"
DEFAULT_RUNS_DIR = ROOT / "runs"


def ensure_layout() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
