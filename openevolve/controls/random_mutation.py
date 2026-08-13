"""Fitness-blind random-mutation control.

Same diff operators and prompts as the main run, but parents/inspirations are
sampled uniformly and every child is kept. Fitness is computed for post-hoc
analysis only and is redacted from the next prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

TRACK_B = Path(__file__).resolve().parent.parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()

from require_review import require_prompt_review  # noqa: E402
import settings  # noqa: E402


def _redact(metrics: dict) -> dict:
    return {k: 0.0 for k in metrics if isinstance(metrics.get(k), (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-review-gate", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "random_mutation")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to run the control without --execute.")
    if not args.skip_review_gate:
        require_prompt_review()

    from patches.parent_lr import apply as apply_parent_lr
    from openevolve.config import load_config
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.prompt.sampler import PromptSampler
    from openevolve.utils.code_utils import apply_diff, extract_diffs
    from evaluator import evaluate

    apply_parent_lr()
    cfg = load_config(TRACK_B / "config.yaml")
    cfg.max_iterations = args.iterations
    llm = LLMEnsemble(cfg.llm.models)
    sampler = PromptSampler(cfg.prompt)
    seed_code = (TRACK_B / "initial_program.py").read_text()
    args.out.mkdir(parents=True, exist_ok=True)
    programs = [
        {
            "id": "seed",
            "code": seed_code,
            "metrics": {"combined_score": 0.0},
            "iteration": 0,
        }
    ]
    rng = random.Random(settings.RANDOM_SEED)
    log = []
    import asyncio

    async def step(i: int) -> None:
        parent = rng.choice(programs)
        inspirations = [p for p in programs if p["id"] != parent["id"]]
        rng.shuffle(inspirations)
        inspirations = inspirations[: cfg.prompt.num_diverse_programs]
        prompt = sampler.build_prompt(
            current_program=parent["code"],
            parent_program=parent["code"],
            program_metrics=_redact(parent["metrics"]),
            previous_programs=[],
            top_programs=[],
            inspirations=[
                {"code": p["code"], "metrics": _redact(p["metrics"]), "id": p["id"]}
                for p in inspirations
            ],
            language="python",
            evolution_round=i,
            diff_based_evolution=True,
        )
        response = await llm.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
        diffs = extract_diffs(response, cfg.diff_pattern)
        if not diffs:
            log.append({"iteration": i, "status": "no_diff"})
            return
        child_code = apply_diff(parent["code"], response, cfg.diff_pattern)
        tmp = args.out / f"child_{i:03d}.py"
        tmp.write_text(child_code)
        result = evaluate(str(tmp))
        metrics = getattr(result, "metrics", result)
        child = {
            "id": str(uuid.uuid4()),
            "code": child_code,
            "metrics": metrics,
            "iteration": i,
            "parent_id": parent["id"],
        }
        programs.append(child)  # keep every child, fitness-blind
        (args.out / f"child_{i:03d}.metrics.json").write_text(json.dumps(metrics, indent=2))
        log.append({"iteration": i, "status": "kept", "id": child["id"], "metrics": metrics})

    for i in range(1, args.iterations + 1):
        print(f"random-mutation iteration {i}/{args.iterations}")
        asyncio.run(step(i))
    (args.out / "log.json").write_text(json.dumps(log, indent=2))
    (args.out / "programs.json").write_text(
        json.dumps([{k: v for k, v in p.items() if k != "code"} for p in programs], indent=2)
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
