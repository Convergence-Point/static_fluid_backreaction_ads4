"""Where generated data lives.

Everything the tools write (solutions, scans, figures) goes under DATA_DIR, which is git-ignored.
Set the environment variable STATIC_ADS_DATA to use another location, e.g. for a scratch run.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("STATIC_ADS_DATA", REPO_ROOT / "data"))

RUNS_DIR = DATA_DIR / "runs"  # one folder per solved star (star.py, sweep.py)
SWEEPS_DIR = DATA_DIR / "sweeps"  # comparison plots of sweeps
SCAN_DIR = DATA_DIR / "scan"  # high-density family scans, profiles and the interactive page
ANALYSIS_DIR = DATA_DIR / "analysis"  # stability analysis of all stored runs
BASIN_DIR = DATA_DIR / "basin"  # Newton basins of the toy problem
STABILITY_DIR = DATA_DIR / "stability"  # stability.py output
FIGURES_DIR = DATA_DIR / "figures"  # figure data for the documentation
TOY_DIR = DATA_DIR / "toy"  # toy-problem iteration histories

DOCS_FIGURES = REPO_ROOT / "docs" / "figures"  # curated figures that are committed
