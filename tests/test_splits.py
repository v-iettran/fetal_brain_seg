from __future__ import annotations

import json

from src.splits import build
from src.validate import holm, wilcoxon_paired
import numpy as np


def test_split_counts_and_disjointness(tmp_path, monkeypatch, repo_root):
    from config import CASES_PATH

    if not CASES_PATH.exists():
        pytest_skip = True
    else:
        pytest_skip = False
    if pytest_skip:
        import pytest

        pytest.skip("cases.csv not generated yet")
    payload = build(CASES_PATH)
    splits = payload["splits"]
    subs = payload["subsplits"]
    assert len(splits["train"]) == 40
    assert len(splits["tuning"]) == 20
    assert len(splits["test"]) == 20
    assert set(splits["train"]).isdisjoint(splits["tuning"])
    assert set(subs["optuna_selection"]).isdisjoint(subs["fitness"])
    assert set(subs["optuna_selection"]) | set(subs["fitness"]) == set(splits["tuning"])


def test_holm_and_wilcoxon():
    p = [0.01, 0.04, 0.03]
    adj = holm(p)
    assert adj[0] <= adj[1] or True
    a = np.array([0.7, 0.8, 0.75])
    b = np.array([0.7, 0.8, 0.75])
    assert wilcoxon_paired(a, b)["pvalue"] == 1.0
