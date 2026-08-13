from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import settings
from harness.guards import load_recipe


@pytest.fixture(scope="module")
def recipe(track_b):
    return load_recipe(track_b / "initial_program.py")


def test_param_count(recipe):
    model = recipe.build_model(1, 8)
    n = sum(p.numel() for p in model.parameters())
    lo, hi = settings.SEED_PARAM_COUNT_RANGE
    assert lo < n < hi, n


def test_forward_shape_and_finite(recipe):
    model = recipe.build_model(1, 8)
    model.eval()
    x = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    logits = y[0] if isinstance(y, (list, tuple)) else y
    assert tuple(logits.shape) == (2, 8, 32, 32, 32)
    assert torch.isfinite(logits).all()


@pytest.mark.gpu
def test_forward_96_cube(recipe):
    model = recipe.build_model(1, 8)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    x = torch.randn(2, 1, 96, 96, 96, device=device)
    model = model.to(device)
    with torch.no_grad():
        y = model(x)
    logits = y[0] if isinstance(y, (list, tuple)) else y
    assert tuple(logits.shape) == (2, 8, 96, 96, 96)
    assert torch.isfinite(logits).all()


def test_initial_cross_entropy(recipe):
    """Random-label CE should be near ln(8). The published seed uses Kaiming on the
    1x1 head, so the value sits a bit above ln(8); a broken class count would be far off.
    """
    torch.manual_seed(0)
    model = recipe.build_model(1, 8)
    model.train()
    x = torch.randn(2, 1, 32, 32, 32)
    target = torch.randint(0, 8, (2, 32, 32, 32))
    loss = float(recipe.build_loss()(model(x), target).detach())
    expected = math.log(8)
    assert 1.5 < loss < 3.0, loss
    assert abs(loss - expected) < 0.5, loss


def test_backward_finite_grads(recipe):
    model = recipe.build_model(1, 8)
    x = torch.randn(1, 1, 32, 32, 32)
    target = torch.randint(0, 8, (1, 32, 32, 32))
    loss = recipe.build_loss()(model(x), target)
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    finite = sum(1 for p in params if p.grad is not None and torch.isfinite(p.grad).all())
    assert finite >= 0.99 * len(params)


@pytest.mark.cuda
def test_peak_vram_under_14gb(recipe):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    model = recipe.build_model(1, 8).to(device)
    x = torch.randn(2, 1, 96, 96, 96, device=device)
    target = torch.randint(0, 8, (2, 96, 96, 96), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    loss = recipe.build_loss()(model(x), target)
    loss.backward()
    gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    assert gb < 14.0, gb


@pytest.mark.gpu
def test_single_patch_overfit(recipe):
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU required")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    torch.manual_seed(0)
    model = recipe.build_model(1, 8).to(device)
    patch = 32 if device.type == "mps" else 96
    x = torch.randn(1, 1, patch, patch, patch, device=device)
    target = torch.zeros(1, patch, patch, patch, dtype=torch.long, device=device)
    target[:, patch // 4 : 3 * patch // 4] = 1
    opt = recipe.build_optimizer(model.parameters(), 1e-3)
    loss_fn = recipe.build_loss()
    model.train()
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x)
        logits = pred[0] if isinstance(pred, (list, tuple)) else pred
        labels = logits.argmax(1)
    dice = (2 * ((labels == 1) & (target == 1)).sum()) / (
        (labels == 1).sum() + (target == 1).sum() + 1e-8
    )
    assert float(dice) > 0.95, float(dice)


def test_augmentation_preserves_labels(recipe):
    rng = np.random.default_rng(0)
    image = np.random.randn(16, 16, 16).astype(np.float32)
    label = np.random.randint(0, 8, (16, 16, 16), dtype=np.int64)
    orig_values = set(label.flatten().tolist())
    aug = recipe.build_augmentation()
    out_i, out_l = aug(image, label, rng)
    assert out_i.shape == image.shape
    assert out_l.shape == label.shape
    assert out_l.dtype == np.int64
    assert set(out_l.flatten().tolist()) <= orig_values
