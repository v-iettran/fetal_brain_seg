"""Sliding-window inference with Gaussian importance weighting. Frozen."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from harness.device import amp_enabled, autocast_dtype, get_device


def gaussian_kernel(patch_size, sigma_div=8.0) -> np.ndarray:
    axes = []
    for p in patch_size:
        sigma = p / sigma_div
        x = np.arange(p) - (p - 1) / 2.0
        axes.append(np.exp(-0.5 * (x / sigma) ** 2))
    kernel = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    kernel = kernel.astype(np.float32)
    kernel /= kernel.max()
    return kernel


def _windows(shape, patch_size, step):
    coords = []
    for size, patch, st in zip(shape, patch_size, step):
        if size <= patch:
            coords.append([0])
            continue
        vals = list(range(0, size - patch + 1, st))
        if vals[-1] != size - patch:
            vals.append(size - patch)
        coords.append(vals)
    for d in coords[0]:
        for h in coords[1]:
            for w in coords[2]:
                yield d, h, w


def predict(model, image_volume, patch_size=(96, 96, 96), overlap=0.5, num_classes=8) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    model.eval()
    image = np.asarray(image_volume, dtype=np.float32)
    step = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    weight = gaussian_kernel(patch_size)
    acc = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    wmap = np.zeros(image.shape, dtype=np.float32)
    use_amp = amp_enabled(device)
    with torch.no_grad():
        for d, h, w in _windows(image.shape, patch_size, step):
            sl = (slice(d, d + patch_size[0]), slice(h, h + patch_size[1]), slice(w, w + patch_size[2]))
            patch = image[sl]
            tensor = torch.from_numpy(patch[None, None].copy()).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype(device), enabled=use_amp):
                out = model(tensor)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            prob = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
            acc[(slice(None),) + sl] += prob * weight
            wmap[sl] += weight
    wmap = np.maximum(wmap, 1e-8)
    pred = np.argmax(acc / wmap, axis=0).astype(np.int64)
    return pred
