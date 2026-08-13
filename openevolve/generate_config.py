"""Generate openevolve/config.yaml from settings.py. Do not hand-edit the YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings  # noqa: E402


def build_config() -> dict:
    system_message = settings.SYSTEM_MESSAGE_PATH.read_text().strip()
    return {
        "max_iterations": settings.MAX_ITERATIONS,
        "random_seed": settings.RANDOM_SEED,
        "checkpoint_interval": settings.CHECKPOINT_INTERVAL,
        "max_tasks_per_child": settings.MAX_TASKS_PER_CHILD,
        "llm": {
            "provider": settings.LLM_PROVIDER,
            "models": list(settings.LLM_MODELS),
            "temperature": settings.LLM_TEMPERATURE,
        },
        "prompt": {
            "num_top_programs": settings.NUM_TOP_PROGRAMS,
            "num_diverse_programs": settings.NUM_DIVERSE_PROGRAMS,
            "include_artifacts": True,
            "system_message": system_message,
        },
        "database": {
            "population_size": settings.POPULATION_SIZE,
            "archive_size": settings.ARCHIVE_SIZE,
            "num_islands": settings.NUM_ISLANDS,
            "migration_interval": settings.MIGRATION_INTERVAL,
            "feature_dimensions": list(settings.FEATURE_DIMENSIONS),
            "feature_bins": dict(settings.FEATURE_BINS),
        },
        "evaluator": {
            "timeout": settings.EVALUATOR_TIMEOUT,
            "parallel_evaluations": settings.PARALLEL_EVALUATIONS,
            "cascade_evaluation": True,
            "cascade_thresholds": list(settings.CASCADE_THRESHOLDS),
            "enable_artifacts": True,
            "use_llm_feedback": False,
            "max_retries": 0,
        },
        "diff_based_evolution": True,
        "max_code_length": settings.MAX_CODE_LENGTH,
    }


def validate_with_openevolve(cfg: dict) -> None:
    try:
        from openevolve.config import Config
    except ImportError:
        print("openevolve not installed; skipping schema validation")
        return
    Config.from_dict(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args()
    cfg = build_config()
    validate_with_openevolve(cfg)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=88))
    print(f"Wrote {args.out} (profile={settings.PROFILE})")


if __name__ == "__main__":
    main()
