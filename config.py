"""Shared project constants. Nothing else hard-codes a path or class name."""

from __future__ import annotations

from pathlib import Path

SEED = 42

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "mri_gz"
CACHE_DIR = REPO_ROOT / "cache" / "npy"
RESULTS_DIR = REPO_ROOT / "results"
SPLITS_PATH = RESULTS_DIR / "splits.json"
CASES_PATH = RESULTS_DIR / "cases.csv"
FINGERPRINT_PATH = RESULTS_DIR / "fingerprint.json"
TRACK_B_SUBSPLITS_PATH = RESULTS_DIR / "trackB_subsplits.json"

NNUNET_RAW_DIR = REPO_ROOT / "nnUNet_raw"
NNUNET_DATASET_NAME = "Dataset001_FeTA"

CLASS_NAMES = {
    0: "background",
    1: "eCSF",
    2: "GM",
    3: "WM",
    4: "ventricles",
    5: "cerebellum",
    6: "deep_GM",
    7: "brainstem",
}

NUM_CLASSES = 8
FOREGROUND_CLASSES = tuple(range(1, NUM_CLASSES))
IN_CHANNELS = 1

MIAL_SUBJECTS = tuple(f"sub-{i:03d}" for i in range(1, 41))
IRTK_SUBJECTS = tuple(f"sub-{i:03d}" for i in range(41, 81))
ALL_SUBJECTS = MIAL_SUBJECTS + IRTK_SUBJECTS

# Shared metric convention when exactly one mask is empty.
HD95_EMPTY_PENALTY_MM = 374.0
