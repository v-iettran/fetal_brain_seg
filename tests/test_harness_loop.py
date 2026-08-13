from __future__ import annotations

import numpy as np
import torch

from harness.data import CaseStore, batch_iterator
from harness.guards import load_recipe
from harness.infer import predict


class TinyStore(CaseStore):
    def __init__(self):
        self.case_ids = ["toy"]
        img = np.zeros((32, 32, 32), dtype=np.float32)
        img[8:24, 8:24, 8:24] = 1.0
        lbl = np.zeros((32, 32, 32), dtype=np.int64)
        lbl[8:24, 8:24, 8:24] = 1
        self._volumes = [(img, lbl)]
        self.cache_dir = None


def test_same_seed_same_batches(track_b):
    recipe = load_recipe(track_b / "initial_program.py")
    store = TinyStore()

    def collect(seed):
        it = batch_iterator(
            store,
            recipe.build_sampler(),
            recipe.build_augmentation(),
            patch_size=(16, 16, 16),
            batch_size=2,
            steps=3,
            seed=seed,
        )
        return [img.numpy().copy() for img, _ in it]

    a = collect(0)
    b = collect(0)
    c = collect(1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert any(not np.array_equal(x, y) for x, y in zip(a, c))


def test_infer_shape_matches(track_b):
    recipe = load_recipe(track_b / "initial_program.py")
    model = recipe.build_model(1, 8)
    model.eval()
    vol = np.random.randn(24, 24, 24).astype(np.float32)
    # Force CPU for this unit test.
    import os

    os.environ["FETA_PROFILE"] = "smoke"
    os.environ["FETA_ALLOW_CPU"] = "1"
    os.environ["FETA_DEVICE"] = "cpu"
    pred = predict(model, vol, patch_size=(16, 16, 16), overlap=0.5)
    assert pred.shape == vol.shape
    assert pred.dtype == np.int64
