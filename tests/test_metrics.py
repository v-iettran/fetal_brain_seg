from __future__ import annotations

import numpy as np

from config import HD95_EMPTY_PENALTY_MM
from src.metrics import dice_coefficient, evaluate_segmentation, hd95_mm, volume_similarity


def _sphere(shape, centre, radius):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= radius ** 2


def test_identical_inputs_perfect_dice():
    gt = np.zeros((16, 16, 16), dtype=np.int64)
    gt[4:12, 4:12, 4:12] = 1
    scores = evaluate_segmentation(gt, gt, (1.0, 1.0, 1.0), classes=(1,))
    assert scores["per_class"][1]["dice"] == 1.0
    assert scores["per_class"][1]["hd95_mm"] == 0.0
    assert scores["per_class"][1]["volume_similarity"] == 1.0


def test_known_overlap_dice():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros_like(a)
    a[:5] = True
    b[3:8] = True
    # intersection 2*10*10=200, sums 500+500=1000, dice=0.4
    assert abs(dice_coefficient(a, b) - 0.4) < 1e-9


def test_empty_prediction_penalty():
    gt = np.zeros((8, 8, 8), dtype=np.int64)
    gt[2:6, 2:6, 2:6] = 1
    pred = np.zeros_like(gt)
    scores = evaluate_segmentation(pred, gt, (0.5, 0.5, 0.5), classes=(1,))
    assert scores["per_class"][1]["dice"] == 0.0
    assert scores["per_class"][1]["volume_similarity"] == 0.0
    assert scores["per_class"][1]["hd95_mm"] == HD95_EMPTY_PENALTY_MM


def test_background_excluded_from_mean():
    gt = np.zeros((8, 8, 8), dtype=np.int64)
    gt[0, 0, 0] = 1
    pred = gt.copy()
    scores = evaluate_segmentation(pred, gt, (1, 1, 1))
    # class 1 is perfect; classes 2-7 are empty in both -> dice 1.0 by convention
    assert scores["mean_dice"] == 1.0


def test_volume_similarity_range():
    a = np.ones((4, 4, 4), dtype=bool)
    b = np.ones((4, 4, 4), dtype=bool)
    b[0] = False
    vs = volume_similarity(a, b)
    assert 0.0 <= vs <= 1.0


def test_hd95_uses_spacing():
    gt = np.zeros((12, 12, 12), dtype=bool)
    pred = np.zeros_like(gt)
    gt[4:8, 4:8, 4:8] = True
    pred[5:9, 4:8, 4:8] = True
    d1 = hd95_mm(pred, gt, (1.0, 1.0, 1.0))
    d2 = hd95_mm(pred, gt, (2.0, 2.0, 2.0))
    assert d2 > d1
