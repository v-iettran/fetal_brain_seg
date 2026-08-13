"""Write SHA-256 manifests for frozen Track B files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

bootstrap.ensure_paths()
from harness.guards import write_frozen_hashes

if __name__ == "__main__":
    payload = write_frozen_hashes()
    for k, v in payload.items():
        print(f"{v}  {k}")
