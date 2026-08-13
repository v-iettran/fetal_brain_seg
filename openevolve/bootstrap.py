"""Put the Track B directory and repo root on sys.path without shadowing the pip openevolve package."""

from __future__ import annotations

import sys
from pathlib import Path

TRACK_B = Path(__file__).resolve().parent
REPO_ROOT = TRACK_B.parent


def ensure_paths() -> None:
    for path in (str(TRACK_B), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
