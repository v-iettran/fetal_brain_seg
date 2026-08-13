"""Case store and guarded batch iterator. Frozen."""

from __future__ import annotations

import time
from typing import Callable, Iterator, Sequence

import numpy as np
import torch

from harness.constants import NUM_CLASSES


class CaseStore:
    def __init__(self, case_ids: Sequence[str], cache_dir: str | Sequence[str] | None = None):
        from pathlib import Path

        if cache_dir is None:
            import settings

            cache_dir = settings.CACHE_DIR
        self.cache_dir = Path(cache_dir)
        self.case_ids = list(case_ids)
        self._volumes: list[tuple[np.ndarray, np.ndarray]] = []
        for sid in self.case_ids:
            img = np.load(self.cache_dir / f"{sid}_img.npy", mmap_mode=None)
            lbl = np.load(self.cache_dir / f"{sid}_lbl.npy", mmap_mode=None)
            if img.dtype != np.float32:
                img = img.astype(np.float32, copy=False)
            if lbl.dtype != np.int64:
                lbl = np.rint(lbl).astype(np.int64)
            self._volumes.append((img, lbl))

    def volumes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self._volumes


def _clip_origin(origin, shape, patch_size):
    clipped = []
    violations = 0
    for o, s, p in zip(origin, shape, patch_size):
        lo = 0
        hi = s - p
        c = int(o)
        if c < lo or c > hi:
            violations += 1
            c = max(lo, min(hi, c))
        clipped.append(c)
    return tuple(clipped), violations


def batch_iterator(
    store: CaseStore,
    sampler,
    augment,
    patch_size=(96, 96, 96),
    batch_size=2,
    steps=4000,
    seed=0,
    augment_timeout_s: float = 2.0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    vols = store.volumes()
    n = len(vols)
    if n == 0:
        raise ValueError("CaseStore is empty")
    origin_violations = 0
    for _ in range(steps):
        images = []
        labels = []
        idxs = rng.integers(0, n, size=batch_size)
        for idx in idxs:
            image, label = vols[int(idx)]
            origin = sampler(label, patch_size, rng)
            origin, v = _clip_origin(origin, label.shape, patch_size)
            origin_violations += v
            sl = tuple(slice(o, o + p) for o, p in zip(origin, patch_size))
            img_p = np.ascontiguousarray(image[sl])
            lbl_p = np.ascontiguousarray(label[sl])
            t0 = time.perf_counter()
            img_p, lbl_p = augment(img_p, lbl_p, rng)
            if time.perf_counter() - t0 > augment_timeout_s:
                raise TimeoutError("augmentation exceeded 2 s/step")
            if img_p.shape != tuple(patch_size) or lbl_p.shape != tuple(patch_size):
                raise ValueError(f"augmented patch shape {img_p.shape}/{lbl_p.shape} != {patch_size}")
            if lbl_p.dtype != np.int64:
                lbl_p = np.rint(lbl_p).astype(np.int64)
            uniq = np.unique(lbl_p)
            if uniq.min() < 0 or uniq.max() >= NUM_CLASSES:
                raise ValueError(f"label values {uniq.tolist()} outside 0..{NUM_CLASSES-1}")
            images.append(img_p)
            labels.append(lbl_p)
        img_t = torch.from_numpy(np.stack(images)[:, None].copy()).float()
        lbl_t = torch.from_numpy(np.stack(labels).copy()).long()
        yield img_t, lbl_t
    # violations are available to callers that inspect the generator; kept for the harness log
    _ = origin_violations
