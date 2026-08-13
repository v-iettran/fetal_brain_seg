from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_B = REPO_ROOT / "openevolve"
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

from bootstrap import ensure_paths  # noqa: E402

ensure_paths()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def track_b() -> Path:
    return TRACK_B
