from __future__ import annotations

import numpy as np

from harness.prepare import crop_to_brain, pad_to_constraints, restore_to_resampled


def test_crop_pad_roundtrip():
    rng = np.random.default_rng(0)
    volume = np.zeros((80, 90, 70), dtype=np.int64)
    volume[20:50, 10:60, 15:55] = rng.integers(0, 8, size=(30, 50, 40))
    image = volume.astype(np.float32)
    image[volume == 0] = 0
    cropped, bbox = crop_to_brain(image)
    cropped_lbl = volume[bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]]
    padded, pads, pre_pad = pad_to_constraints(cropped_lbl)
    restored = restore_to_resampled(padded, bbox, volume.shape, pads, pre_pad)
    assert np.array_equal(restored, volume)
