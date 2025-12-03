"""Project-wide configuration constants."""
from __future__ import annotations

from typing import Final
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR: Final[str] = os.path.join(ROOT, 'results')
RANDOM_SEED: Final[int] = 42
ROLLING_WINDOW: Final[int] = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
