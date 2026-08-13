"""Pure segmentation metrics. No I/O, no plotting, no global state.

Conventions
-----------
* Scores are computed per class, then averaged over classes 1–7. Background
  (class 0) is excluded from the mean.
* If a class exists in the ground truth but the prediction has zero voxels of
  it: Dice = 0, volume similarity = 0, HD95 = HD95_EMPTY_PENALTY_MM (100.0 mm).
* HD95 uses the real voxel spacing so values are in millimetres.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from config import FOREGROUND_CLASSES, HD95_EMPTY_PENALTY_MM, NUM_CLASSES


def _as_int_labels(arr: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(arr)).astype(np.int64, copy=False)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def volume_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_v = float(pred.astype(bool).sum())
    gt_v = float(gt.astype(bool).sum())
    if pred_v + gt_v == 0:
        return 1.0
    return float(1.0 - abs(pred_v - gt_v) / (pred_v + gt_v))


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.int64)
    struct = ndi.generate_binary_structure(3, 1)
    eroded = ndi.binary_erosion(mask, structure=struct, border_value=0)
    surface = mask & ~eroded
    return np.argwhere(surface)


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]) -> float:
    """95th-percentile symmetric Hausdorff distance in millimetres."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(HD95_EMPTY_PENALTY_MM)
    spacing = np.asarray(spacing, dtype=np.float64)
    dt_pred = ndi.distance_transform_edt(~pred, sampling=spacing)
    dt_gt = ndi.distance_transform_edt(~gt, sampling=spacing)
    pred_surf = _surface_voxels(pred)
    gt_surf = _surface_voxels(gt)
    d_pred_to_gt = dt_gt[tuple(pred_surf.T)]
    d_gt_to_pred = dt_pred[tuple(gt_surf.T)]
    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(distances, 95))


def euler_characteristic(mask: np.ndarray) -> int:
    """3D Euler number via 26-connected components minus interior cavities."""
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return 0
    _, n_obj = ndi.label(mask, structure=ndi.generate_binary_structure(3, 3))
    inv = ~mask
    labeled_inv, n_inv = ndi.label(inv, structure=ndi.generate_binary_structure(3, 1))
    # Components of the inverse that touch the volume border are exterior background.
    border = np.zeros_like(inv, dtype=bool)
    border[[0, -1], :, :] = True
    border[:, [0, -1], :] = True
    border[:, :, [0, -1]] = True
    border_labels = set(int(v) for v in np.unique(labeled_inv[border]) if v != 0)
    n_cavities = n_inv - len(border_labels)
    return int(n_obj - n_cavities)


def score_class(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_id: int,
    voxel_spacing: Sequence[float],
) -> dict[str, float]:
    pred = _as_int_labels(prediction) == class_id
    gt = _as_int_labels(ground_truth) == class_id
    gt_present = bool(gt.any())
    pred_present = bool(pred.any())
    if gt_present and not pred_present:
        return {
            "dice": 0.0,
            "hd95_mm": float(HD95_EMPTY_PENALTY_MM),
            "volume_similarity": 0.0,
            "euler_diff": float(abs(euler_characteristic(gt))),
            "empty_prediction": 1.0,
        }
    return {
        "dice": dice_coefficient(pred, gt),
        "hd95_mm": hd95_mm(pred, gt, voxel_spacing),
        "volume_similarity": volume_similarity(pred, gt),
        "euler_diff": float(abs(euler_characteristic(pred) - euler_characteristic(gt))),
        "empty_prediction": 0.0 if pred_present or not gt_present else 1.0,
    }


def evaluate_segmentation(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    voxel_spacing: Sequence[float],
    classes: Sequence[int] = FOREGROUND_CLASSES,
) -> dict:
    """Return per-class and mean scores. Background is excluded from the mean."""
    prediction = _as_int_labels(prediction)
    ground_truth = _as_int_labels(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"shape mismatch: pred {prediction.shape} vs gt {ground_truth.shape}")
    per_class = {}
    for c in classes:
        per_class[int(c)] = score_class(prediction, ground_truth, int(c), voxel_spacing)
    mean_dice = float(np.mean([per_class[c]["dice"] for c in classes]))
    mean_hd95 = float(np.mean([per_class[c]["hd95_mm"] for c in classes]))
    mean_vs = float(np.mean([per_class[c]["volume_similarity"] for c in classes]))
    return {
        "mean_dice": mean_dice,
        "mean_hd95_mm": mean_hd95,
        "mean_volume_similarity": mean_vs,
        "worst_class_dice": float(min(per_class[c]["dice"] for c in classes)),
        "per_class": per_class,
        "hd95_empty_penalty_mm": float(HD95_EMPTY_PENALTY_MM),
        "n_classes_scored": int(len(classes)),
        "num_classes_in_label_space": int(NUM_CLASSES),
    }


def mean_foreground_dice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    voxel_spacing: Sequence[float] | None = None,
) -> float:
    spacing = voxel_spacing if voxel_spacing is not None else (1.0, 1.0, 1.0)
    return float(evaluate_segmentation(prediction, ground_truth, spacing)["mean_dice"])
