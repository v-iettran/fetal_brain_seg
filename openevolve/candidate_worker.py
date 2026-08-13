"""Subprocess entry point for one evaluation stage. Writes JSON to --result."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

TRACK_B = Path(__file__).resolve().parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--stage", required=True, choices=["stage1", "stage2", "smoke_train"])
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    try:
        from evaluator_impl import run_stage

        payload = run_stage(args.program, args.stage)
    except Exception as exc:
        payload = {
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
    Path(args.result).write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
