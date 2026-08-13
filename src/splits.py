"""Deterministic 40/20/20 split plus Track B 10/10 tuning subdivision.

Balance is mandatory on reconstruction method and recommended on ICV tertile.
This file writes results/splits.json and results/trackB_subsplits.json once.
Regenerating is forbidden after the pre-registration commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CASES_PATH,
    RESULTS_DIR,
    SEED,
    SPLITS_PATH,
    TRACK_B_SUBSPLITS_PATH,
)


def _load_cases(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["icv_mm3"] = float(r["icv_mm3"])
    return rows


def _tertiles(rows: list[dict]) -> dict[str, int]:
    ordered = sorted(rows, key=lambda r: (r["icv_mm3"], r["subject_id"]))
    n = len(ordered)
    cuts = [n // 3, (2 * n) // 3]
    out = {}
    for i, r in enumerate(ordered):
        if i < cuts[0]:
            out[r["subject_id"]] = 0
        elif i < cuts[1]:
            out[r["subject_id"]] = 1
        else:
            out[r["subject_id"]] = 2
    return out


def _allocate(ids: list[str], n_train: int, n_tune: int, n_test: int, rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    ids = list(ids)
    rng.shuffle(ids)
    train = ids[:n_train]
    tune = ids[n_train : n_train + n_tune]
    test = ids[n_train + n_tune : n_train + n_tune + n_test]
    leftover = n_train + n_tune + n_test - (len(train) + len(tune) + len(test))
    if leftover != 0 or len(train) + len(tune) + len(test) != len(ids):
        # Exact partition of the whole list.
        train = ids[:n_train]
        tune = ids[n_train : n_train + n_tune]
        test = ids[n_train + n_tune :]
    return train, tune, test


def _stratified_split(rows: list[dict], rng: random.Random) -> dict[str, list[str]]:
    """Split one reconstruction group of 40 into 20/10/10, balanced on ICV tertile."""
    tertile = _tertiles(rows)
    buckets: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        buckets[tertile[r["subject_id"]]].append(r["subject_id"])
    train, tune, test = [], [], []
    # 40 subjects, 3 tertiles of 13/13/14. Target 20/10/10 overall.
    # Per tertile of size n: train = round(n/2) with leftover going to train first,
    # then tune/test split the remainder equally.
    remainder_pool = []
    for t in (0, 1, 2):
        ids = sorted(buckets[t])
        n = len(ids)
        n_train = n // 2
        rest = n - n_train
        n_tune = rest // 2
        n_test = rest - n_tune
        tr, tu, te = _allocate(ids, n_train, n_tune, n_test, rng)
        train.extend(tr)
        tune.extend(tu)
        test.extend(te)
        remainder_pool.extend([])
    # Fix any drift from integer rounding so totals are exactly 20/10/10.
    train, tune, test = [sorted(x) for x in (train, tune, test)]
    def _move(src, dst, n):
        rng.shuffle(src)
        moved = src[:n]
        del src[:n]
        dst.extend(moved)
    while len(train) > 20:
        _move(train, tune if len(tune) < 10 else test, 1)
    while len(tune) > 10:
        _move(tune, test if len(test) < 10 else train, 1)
    while len(test) > 10:
        _move(test, train if len(train) < 20 else tune, 1)
    while len(train) < 20:
        _move(tune if len(tune) > 10 else test, train, 1)
    while len(tune) < 10:
        _move(train if len(train) > 20 else test, tune, 1)
    while len(test) < 10:
        _move(train if len(train) > 20 else tune, test, 1)
    return {
        "train": sorted(train),
        "tuning": sorted(tune),
        "test": sorted(test),
        "icv_tertile": tertile,
    }


def _subdivide_tuning(tune_rows: list[dict], rng: random.Random) -> tuple[list[str], list[str]]:
    """10/10 from 20 tuning cases: 5 mial + 5 irtk each, ICV-balanced within rec."""
    optuna, fitness = [], []
    for rec in ("mial", "irtk"):
        rec_rows = [r for r in tune_rows if r["reconstruction"] == rec]
        tertile = _tertiles(rec_rows)
        buckets: dict[int, list[str]] = defaultdict(list)
        for r in rec_rows:
            buckets[tertile[r["subject_id"]]].append(r["subject_id"])
        chosen_optuna = []
        for t in (0, 1, 2):
            ids = sorted(buckets[t])
            rng.shuffle(ids)
            # Aim for ~1–2 per tertile so 5 total.
            take = 1 if ids else 0
            chosen_optuna.extend(ids[:take])
            buckets[t] = ids[take:]
        leftover = []
        for t in (0, 1, 2):
            leftover.extend(buckets[t])
        rng.shuffle(leftover)
        while len(chosen_optuna) < 5 and leftover:
            chosen_optuna.append(leftover.pop())
        rec_ids = {r["subject_id"] for r in rec_rows}
        rec_fitness = sorted(rec_ids - set(chosen_optuna))
        optuna.extend(sorted(chosen_optuna)[:5])
        fitness.extend(rec_fitness[:5])
    return sorted(optuna), sorted(fitness)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build(cases_path: Path = CASES_PATH) -> dict:
    rows = _load_cases(cases_path)
    if len(rows) != 80:
        raise ValueError(f"Expected 80 subjects in {cases_path}, got {len(rows)}")
    rng = random.Random(SEED)
    mial = [r for r in rows if r["reconstruction"] == "mial"]
    irtk = [r for r in rows if r["reconstruction"] == "irtk"]
    if len(mial) != 40 or len(irtk) != 40:
        raise ValueError(f"Expected 40 mial + 40 irtk, got {len(mial)}/{len(irtk)}")
    mial_split = _stratified_split(mial, rng)
    irtk_split = _stratified_split(irtk, rng)
    splits = {
        "seed": SEED,
        "train": sorted(mial_split["train"] + irtk_split["train"]),
        "tuning": sorted(mial_split["tuning"] + irtk_split["tuning"]),
        "test": sorted(mial_split["test"] + irtk_split["test"]),
        "by_reconstruction": {
            "mial": {k: mial_split[k] for k in ("train", "tuning", "test")},
            "irtk": {k: irtk_split[k] for k in ("train", "tuning", "test")},
        },
        "icv_tertile": {**mial_split["icv_tertile"], **irtk_split["icv_tertile"]},
        "notes": (
            "Gestational age and pathology labels were unavailable. ICV derived "
            "from the reference segmentation is the maturity proxy used for tertiles."
        ),
    }
    assert len(splits["train"]) == 40
    assert len(splits["tuning"]) == 20
    assert len(splits["test"]) == 20
    assert set(splits["train"]).isdisjoint(splits["tuning"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["tuning"]).isdisjoint(splits["test"])
    for rec in ("mial", "irtk"):
        br = splits["by_reconstruction"][rec]
        assert len(br["train"]) == 20 and len(br["tuning"]) == 10 and len(br["test"]) == 10

    tune_rows = [r for r in rows if r["subject_id"] in set(splits["tuning"])]
    optuna_ids, fitness_ids = _subdivide_tuning(tune_rows, rng)
    subsplits = {
        "seed": SEED,
        "proxy_train": splits["train"],
        "optuna_selection": optuna_ids,
        "fitness": fitness_ids,
        "sealed_test": splits["test"],
    }
    assert len(optuna_ids) == 10 and len(fitness_ids) == 10
    assert set(optuna_ids).isdisjoint(fitness_ids)
    assert set(optuna_ids) | set(fitness_ids) == set(splits["tuning"])
    for rec, rec_ids in (
        ("mial", set(splits["by_reconstruction"]["mial"]["tuning"])),
        ("irtk", set(splits["by_reconstruction"]["irtk"]["tuning"])),
    ):
        assert sum(1 for s in optuna_ids if s in rec_ids) == 5
        assert sum(1 for s in fitness_ids if s in rec_ids) == 5
    return {"splits": splits, "subsplits": subsplits}


def write(force: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SPLITS_PATH.exists() and not force:
        raise SystemExit(
            f"{SPLITS_PATH} already exists. Refusing to regenerate. Pass --force only "
            "before the pre-registration commit."
        )
    if not CASES_PATH.exists():
        from src.explore import run as explore_run

        explore_run()
    payload = build()
    SPLITS_PATH.write_text(json.dumps(payload["splits"], indent=2) + "\n")
    TRACK_B_SUBSPLITS_PATH.write_text(json.dumps(payload["subsplits"], indent=2) + "\n")
    print(f"Wrote {SPLITS_PATH}")
    print(f"  sha256={file_sha256(SPLITS_PATH)}")
    print(f"Wrote {TRACK_B_SUBSPLITS_PATH}")
    print(f"  sha256={file_sha256(TRACK_B_SUBSPLITS_PATH)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate immutable FeTA splits")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits (pre-registration only)")
    args = parser.parse_args()
    write(force=args.force)


if __name__ == "__main__":
    main()
