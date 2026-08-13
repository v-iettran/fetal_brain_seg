"""Generic training loop over the six-function contract. Frozen."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from harness.data import CaseStore, batch_iterator
from harness.device import amp_enabled, autocast_dtype, get_device


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    recipe,
    case_ids,
    lr,
    total_steps,
    seed,
    cache_dir,
    patch_size=(96, 96, 96),
    batch_size=2,
    grad_clip_norm=12.0,
    nan_abort_steps=50,
    augment_timeout_s=2.0,
) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    set_seeds(seed)
    device = get_device()
    model = recipe.build_model(1, 8).to(device)
    loss_fn = recipe.build_loss()
    opt = recipe.build_optimizer(model.parameters(), lr)
    sched = recipe.build_scheduler(opt, total_steps)
    store = CaseStore(case_ids, cache_dir)
    sampler = recipe.build_sampler()
    augment = recipe.build_augmentation()
    batches = batch_iterator(
        store,
        sampler,
        augment,
        patch_size=patch_size,
        batch_size=batch_size,
        steps=total_steps,
        seed=seed,
        augment_timeout_s=augment_timeout_s,
    )
    use_amp = amp_enabled(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = []
    nan_run = 0
    model.train()
    for step, (images, labels) in enumerate(batches, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype(device), enabled=use_amp):
            output = model(images)
            loss = loss_fn(output, labels)
        if not torch.isfinite(loss):
            nan_run += 1
            if nan_run >= nan_abort_steps:
                return None, {"reason": "nan_loss", "step": step, "losses": losses}
            continue
        nan_run = 0
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        if step % 100 == 0 or step == 1:
            losses.append({"step": step, "loss": float(loss.detach().cpu())})
    return model, {"losses": losses, "steps": total_steps, "lr": float(lr)}
