"""Apply the version-checked parent_lr patch and print its hash."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patches.parent_lr import REQUIRED_VERSION, apply, sha256  # noqa: E402


def main() -> None:
    apply()
    print(f"Applied parent_lr patch for OpenEvolve {REQUIRED_VERSION}")
    print(f"sha256={sha256()}")


if __name__ == "__main__":
    main()
