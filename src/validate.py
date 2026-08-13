"""Evaluation harness and paired statistical tests.

The sealed test split is blocked until results/sealed_test_unlock.json exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CASES_PATH,
    FOREGROUND_CLASSES,
    RESULTS_DIR,
    SPLITS_PATH,
    TRACK_B_SUBSPLITS_PATH,
)
from src.metrics import evaluate_segmentation  # noqa: E402

SEALED_UNLOCK = RESULTS_DIR / "sealed_test_unlock.json"
SEALED_AUDIT = RESULTS_DIR / "sealed_test_audit.jsonl"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def assert_not_sealed(case_ids: Iterable[str]) -> None:
    splits = _load_json(SPLITS_PATH)
    sealed = set(splits["test"])
    hit = sorted(set(case_ids) & sealed)
    if not hit:
        return
    if not SEALED_UNLOCK.exists():
        raise PermissionError(
            "Attempted to score sealed test cases without unlock file "
            f"{SEALED_UNLOCK}: {hit}"
        )
    unlock = _load_json(SEALED_UNLOCK)
    if unlock.get("confirm") != "OPEN_SEALED_TEST":
        raise PermissionError(f"{SEALED_UNLOCK} is missing confirm=OPEN_SEALED_TEST")
    SEALED_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with SEALED_AUDIT.open("a") as f:
        f.write(json.dumps({"cases": hit, "reason": unlock.get("reason", "")}) + "\n")


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adj[idx] = running
    return adj.tolist()


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    if np.allclose(a, b):
        return {"statistic": 0.0, "pvalue": 1.0}
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}
    except ValueError:
        return {"statistic": float("nan"), "pvalue": 1.0}


def bootstrap_ranking(scores_by_config: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    names = list(scores_by_config)
    n = len(next(iter(scores_by_config.values())))
    wins = {name: 0 for name in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means = {name: float(scores_by_config[name][idx].mean()) for name in names}
        winner = max(means, key=means.get)
        wins[winner] += 1
    return {name: wins[name] / n_boot for name in names}


def compare_configs(per_subject: dict[str, dict[str, dict[int, float]]]) -> dict:
    """per_subject[config][subject_id][class_id] = dice."""
    configs = list(per_subject)
    if len(configs) < 2:
        return {}
    subjects = sorted(set.intersection(*[set(per_subject[c]) for c in configs]))
    out = {}
    for i, a in enumerate(configs):
        for b in configs[i + 1 :]:
            pvals = []
            per_class = {}
            for c in FOREGROUND_CLASSES:
                va = np.array([per_subject[a][s][c] for s in subjects], dtype=np.float64)
                vb = np.array([per_subject[b][s][c] for s in subjects], dtype=np.float64)
                test = wilcoxon_paired(va, vb)
                per_class[int(c)] = test
                pvals.append(test["pvalue"])
            out[f"{a}_vs_{b}"] = {
                "per_class": per_class,
                "holm_pvalues": {int(c): p for c, p in zip(FOREGROUND_CLASSES, holm(pvals))},
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a folder of predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["tuning", "fitness", "optuna_selection", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = _load_json(SPLITS_PATH)
    subs = _load_json(TRACK_B_SUBSPLITS_PATH)
    mapping = {
        "tuning": splits["tuning"],
        "test": splits["test"],
        "fitness": subs["fitness"],
        "optuna_selection": subs["optuna_selection"],
    }
    case_ids = mapping[args.split]
    assert_not_sealed(case_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"split": args.split, "n": len(case_ids), "status": "predictions_not_scored_here"}, indent=2)
    )


if __name__ == "__main__":
    main()
