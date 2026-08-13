"""Launch OpenEvolve with the parent_lr compatibility patch applied."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TRACK_B = Path(__file__).resolve().parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()

from patches.parent_lr import apply as apply_parent_lr  # noqa: E402
from require_review import require_prompt_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Track B OpenEvolve")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-review-gate", action="store_true")
    parser.add_argument("--config", default=str(TRACK_B / "config.yaml"))
    args = parser.parse_args()
    if not args.skip_review_gate:
        require_prompt_review()
    apply_parent_lr()
    from openevolve.cli import main_async
    import asyncio

    argv = [
        str(TRACK_B / "initial_program.py"),
        str(TRACK_B / "evaluator.py"),
        "--config",
        args.config,
    ]
    if args.output:
        argv += ["--output", args.output]
    else:
        import settings

        argv += ["--output", str(settings.OUTPUT_DIR)]
    if args.iterations is not None:
        argv += ["--iterations", str(args.iterations)]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    sys.argv = ["openevolve-run", *argv]
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
