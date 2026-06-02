"""Canonical filesystem layout for the project.

Single source of truth for where runs, results, cache and figures live, so every
script (experiments, generators, apps) resolves the same absolute paths regardless
of the current working directory. Import these instead of hardcoding relative dirs.

    from paths import RUNS_DIR, FIGURES_DIR, RESULTS_DIR, CACHE_DIR
"""
from __future__ import annotations

from pathlib import Path

# Experiments/src/paths.py -> parent = src/ -> parent.parent = Experiments/
EXP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_ROOT.parent

RUNS_DIR = EXP_ROOT / "runs"          # neuroevolution run npz + pre-generated datasets
RESULTS_DIR = EXP_ROOT / "results"    # csv metrics
CACHE_DIR = RESULTS_DIR / "cache"     # cached embeddings (npz)
FIGURES_DIR = EXP_ROOT / "figures"    # generated figures
PAPER_FIGURES = REPO_ROOT / "Docs" / "papers" / "figures"


def ensure_dirs() -> None:
    for d in (RUNS_DIR, RESULTS_DIR, CACHE_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
