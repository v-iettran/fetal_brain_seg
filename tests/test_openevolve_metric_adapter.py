from __future__ import annotations

import numpy as np
import pytest

from evaluator_impl import shared_dice_summary
from src.metrics import compute_metrics


def test_shared_metric_adapter_preserves_mean_and_class_order() -> None:
    prediction = np.zeros((4, 4, 4), dtype=np.uint8)
    ground_truth = np.zeros_like(prediction)
    for class_id in range(1, 8):
        prediction[class_id // 4, class_id % 4, 1] = class_id
        ground_truth[class_id // 4, class_id % 4, 1] = class_id
    prediction[0, 1, 1] = 0

    spacing = (0.5, 0.5, 0.5)
    shared = compute_metrics(prediction, ground_truth, spacing)
    mean_dice, per_class_dice = shared_dice_summary(prediction, ground_truth, spacing)

    assert mean_dice == pytest.approx(shared["mean"]["dice"])
    assert per_class_dice == pytest.approx(
        [shared[class_id]["dice"] for class_id in range(1, 8)]
    )
