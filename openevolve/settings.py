"""Single knob file for Track B. Hand-edit this; do not hand-edit config.yaml.

Pre-registered values (splits, fitness metric, target list) live in harness/
and PREREGISTRATION.md, not here.

Changing this file after iteration 1 invalidates the run.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Profile. "production" is the scientific experiment. "smoke" is local MPS
# debugging and is not comparable.
# ---------------------------------------------------------------------------
PROFILE = os.environ.get("FETA_PROFILE", "production")
if PROFILE not in {"production", "smoke"}:
    raise ValueError(f"FETA_PROFILE must be production or smoke, got {PROFILE!r}")

# ---------------------------------------------------------------------------
# Search budget
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 150
CHECKPOINT_INTERVAL = 10
RANDOM_SEED = 42
POPULATION_SIZE = 120
NUM_ISLANDS = 3
MIGRATION_INTERVAL = 25
ARCHIVE_SIZE = 36
FEATURE_DIMENSIONS = ["params_millions", "worst_class_dice"]
FEATURE_BINS = {"params_millions": 6, "worst_class_dice": 6}

# ---------------------------------------------------------------------------
# LLM (Claude Code CLI). temperature/max_tokens are recorded for provenance;
# the Claude Code backend does not honor them.
# ---------------------------------------------------------------------------
LLM_PROVIDER = "claude_code"
LLM_TEMPERATURE = 0.8
LLM_MODELS = [
    {"name": "sonnet", "weight": 0.8, "max_tokens": 16000, "max_budget_usd": 40.0, "timeout": 300},
    {"name": "opus", "weight": 0.2, "max_tokens": 16000, "max_budget_usd": 30.0, "timeout": 300},
]
NUM_TOP_PROGRAMS = 3
NUM_DIVERSE_PROGRAMS = 2

# ---------------------------------------------------------------------------
# Proxy training
# ---------------------------------------------------------------------------
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 2
PROXY_STEPS = 4000
GRAD_CLIP_NORM = 12.0
NAN_ABORT_STEPS = 50
OVERLAP = 0.5
INFER_SIGMA_DIV = 8.0

# ---------------------------------------------------------------------------
# Inner Optuna. Declared as (name, distribution, low, high) so evaluator.py
# does not hard-code the search space.
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = 3
OPTUNA_PRUNER = "none"
OPTUNA_SAMPLER = "tpe"
SEARCHABLE_PARAMS = [
    ("lr", "log_uniform", 1e-4, 1e-1),
]
DEFAULT_LR = 1e-3
REEVAL_SEED = 7777
TRIAL_SEED_BASE = 1000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = ROOT / "cache" / "npy"
MRI_DIR = ROOT / "mri_gz"
OUTPUT_DIR = ROOT / "results" / "openevolve_output"
SPLITS_PATH = ROOT / "results" / "splits.json"
SUBSPLITS_PATH = ROOT / "results" / "trackB_subsplits.json"
SYSTEM_MESSAGE_PATH = Path(__file__).resolve().parent / "prompts" / "system_message.txt"

# ---------------------------------------------------------------------------
# Concurrency / evaluator
# ---------------------------------------------------------------------------
EVALUATOR_TIMEOUT = 7200
PARALLEL_EVALUATIONS = 3
CASCADE_THRESHOLDS = [0.35]
MAX_CODE_LENGTH = 20000
MAX_TASKS_PER_CHILD = 1
AUGMENT_TIMEOUT_S = 2.0
PARAM_COUNT_RANGE = (1e6, 1.5e8)
STAGE1_VRAM_GB = 14.0
SEED_PARAM_COUNT_RANGE = (18.5e6, 19.5e6)

# ---------------------------------------------------------------------------
# Smoke overrides — not scientific
# ---------------------------------------------------------------------------
if PROFILE == "smoke":
    MAX_ITERATIONS = 2
    CHECKPOINT_INTERVAL = 1
    POPULATION_SIZE = 8
    NUM_ISLANDS = 2
    MIGRATION_INTERVAL = 10
    PROXY_STEPS = 4
    BATCH_SIZE = 1
    OPTUNA_N_TRIALS = 1
    PARALLEL_EVALUATIONS = 1
    EVALUATOR_TIMEOUT = 600
    OUTPUT_DIR = ROOT / "results" / "openevolve_output" / "smoke"
    LLM_MODELS = [
        {"name": "sonnet", "weight": 1.0, "max_tokens": 4000, "max_budget_usd": 1.0, "timeout": 120},
    ]

# ---------------------------------------------------------------------------
# Validation on import
# ---------------------------------------------------------------------------
assert all(p % 8 == 0 for p in PATCH_SIZE), "patch size must be divisible by 8"
assert OPTUNA_N_TRIALS >= 1
assert NUM_ISLANDS >= 1
_wsum = sum(m["weight"] for m in LLM_MODELS)
assert abs(_wsum - 1.0) < 1e-6, f"LLM weights must sum to 1, got {_wsum}"
assert BATCH_SIZE >= 1
assert PROXY_STEPS >= 1
assert 0.0 < OVERLAP < 1.0


def as_dict() -> dict:
    return {
        "profile": PROFILE,
        "max_iterations": MAX_ITERATIONS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "random_seed": RANDOM_SEED,
        "population_size": POPULATION_SIZE,
        "num_islands": NUM_ISLANDS,
        "migration_interval": MIGRATION_INTERVAL,
        "archive_size": ARCHIVE_SIZE,
        "feature_dimensions": list(FEATURE_DIMENSIONS),
        "feature_bins": dict(FEATURE_BINS),
        "llm_provider": LLM_PROVIDER,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_models": list(LLM_MODELS),
        "num_top_programs": NUM_TOP_PROGRAMS,
        "num_diverse_programs": NUM_DIVERSE_PROGRAMS,
        "patch_size": list(PATCH_SIZE),
        "batch_size": BATCH_SIZE,
        "proxy_steps": PROXY_STEPS,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "optuna_n_trials": OPTUNA_N_TRIALS,
        "optuna_pruner": OPTUNA_PRUNER,
        "searchable_params": [list(x) for x in SEARCHABLE_PARAMS],
        "default_lr": DEFAULT_LR,
        "reeval_seed": REEVAL_SEED,
        "cache_dir": str(CACHE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "evaluator_timeout": EVALUATOR_TIMEOUT,
        "parallel_evaluations": PARALLEL_EVALUATIONS,
        "cascade_thresholds": list(CASCADE_THRESHOLDS),
        "max_code_length": MAX_CODE_LENGTH,
        "max_tasks_per_child": MAX_TASKS_PER_CHILD,
    }
