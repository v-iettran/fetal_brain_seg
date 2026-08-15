"""Stage 1/2 evaluation logic. Imported only inside the isolated worker."""

from __future__ import annotations

import inspect
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch

import settings
from harness import guards
from harness.constants import IN_CHANNELS, NUM_CLASSES
from harness.device import get_device
from harness.infer import predict
from harness.train_loop import train
from src.metrics import compute_metrics

SEED_PATH = Path(__file__).resolve().parent / "initial_program.py"
PARENT_SIDECAR_ENV = "FETA_PARENT_SIDECAR"


def _fail(reason: str, **extra) -> dict:
    return {"ok": False, "reason": reason, **extra}


def _load_splits() -> tuple[dict, dict]:
    splits = json.loads(settings.SPLITS_PATH.read_text())
    subs = json.loads(settings.SUBSPLITS_PATH.read_text())
    guards.assert_split_integrity(subs, splits)
    return splits, subs


def _parent_best_lr(program_path: str) -> float | None:
    sidecar = os.environ.get(PARENT_SIDECAR_ENV)
    candidates = []
    if sidecar:
        candidates.append(Path(sidecar))
    candidates.append(Path(program_path + ".parent.json"))
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            lr = data.get("best_lr")
            if lr is not None:
                return float(lr)
    return None


def shared_dice_summary(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    voxel_spacing: tuple[float, float, float],
) -> tuple[float, list[float]]:
    """Adapt the shared Track A metric output to the Track B fitness contract."""
    scores = compute_metrics(prediction, ground_truth, voxel_spacing)
    mean_dice = float(scores["mean"]["dice"])
    per_class_dice = [float(scores[c]["dice"]) for c in range(1, NUM_CLASSES)]
    return mean_dice, per_class_dice


def _suggest_params(trial, parent_lr: float | None) -> dict:
    values = {}
    for name, dist, low, high in settings.SEARCHABLE_PARAMS:
        if dist == "log_uniform":
            values[name] = trial.suggest_float(name, float(low), float(high), log=True)
        elif dist == "uniform":
            values[name] = trial.suggest_float(name, float(low), float(high))
        else:
            raise ValueError(f"unknown distribution {dist}")
    return values


def run_stage1(program_path: str) -> dict:
    guards.verify_frozen_files()
    guards.check_imports(program_path)
    guards.check_bare_except(program_path)
    guards.check_forbidden_calls(program_path)
    guards.frozen_prefix_suffix(SEED_PATH, Path(program_path))
    recipe = guards.load_recipe(program_path)
    guards.check_contract(recipe)

    device = get_device()
    model = recipe.build_model(IN_CHANNELS, NUM_CLASSES).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    lo, hi = settings.PARAM_COUNT_RANGE
    if not (lo <= n_params <= hi):
        return _fail(f"param count {n_params} outside [{lo}, {hi}]")

    x = torch.randn(2, 1, *settings.PATCH_SIZE, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = model(x)
    logits = out[0] if isinstance(out, (list, tuple)) else out
    if tuple(logits.shape) != (2, NUM_CLASSES, *settings.PATCH_SIZE):
        return _fail(f"bad output shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        return _fail("non-finite forward output")

    target = torch.randint(0, NUM_CLASSES, (2, *settings.PATCH_SIZE), device=device)
    loss_fn = recipe.build_loss()
    loss = loss_fn(out, target)
    if loss.ndim != 0 or not torch.isfinite(loss):
        return _fail(f"loss is not a finite scalar: {loss}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    finite = sum(1 for g in grads if g is not None and torch.isfinite(g).all())
    if finite < 0.9 * max(len(grads), 1):
        return _fail("fewer than 90% of parameters have finite gradients")

    vram_gb = 0.0
    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if vram_gb >= settings.STAGE1_VRAM_GB:
            return _fail(f"peak VRAM {vram_gb:.2f} GB >= {settings.STAGE1_VRAM_GB}")

    opt = recipe.build_optimizer(model.parameters(), settings.DEFAULT_LR)
    _ = recipe.build_scheduler(opt, settings.PROXY_STEPS)
    _ = recipe.build_sampler()
    _ = recipe.build_augmentation()
    return {
        "ok": True,
        "stage1_ok": 1.0,
        "params_millions": n_params / 1e6,
        "n_params": n_params,
        "vram_gb": float(vram_gb),
    }


def _score_cases(model, case_ids, cache_dir) -> tuple[float, list[float], str | None]:
    from harness.data import CaseStore

    store = CaseStore(case_ids, cache_dir)
    dices = []
    per_class_acc = [[] for _ in range(7)]
    for (image, label), sid in zip(store.volumes(), case_ids):
        pred = predict(model, image, patch_size=settings.PATCH_SIZE, overlap=settings.OVERLAP)
        try:
            guards.assert_prediction_sane(pred, reference_shape=image.shape)
        except AssertionError as exc:
            return 0.0, [0.0] * 7, f"{sid}: {exc}"
        mean_dice, per_class_dice = shared_dice_summary(pred, label, (0.5, 0.5, 0.5))
        dices.append(mean_dice)
        for class_scores, score in zip(per_class_acc, per_class_dice):
            class_scores.append(score)
    per_class = [float(np.mean(xs)) for xs in per_class_acc]
    return float(np.mean(dices)), per_class, None


def run_stage2(program_path: str) -> dict:
    stage1 = run_stage1(program_path)
    if not stage1.get("ok"):
        return stage1
    guards.verify_frozen_files()
    _, subs = _load_splits()
    recipe = guards.load_recipe(program_path)
    parent_lr = _parent_best_lr(program_path)

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=settings.RANDOM_SEED)
    pruner = (
        optuna.pruners.MedianPruner()
        if settings.OPTUNA_PRUNER == "median"
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    if parent_lr is not None:
        study.enqueue_trial({"lr": float(parent_lr)})

    cache_dir = settings.CACHE_DIR
    proxy_ids = subs["proxy_train"]
    optuna_ids = subs["optuna_selection"]
    fitness_ids = subs["fitness"]
    if settings.PROFILE == "smoke":
        proxy_ids = proxy_ids[:2]
        optuna_ids = optuna_ids[:1]
        fitness_ids = fitness_ids[:1]

    def objective(trial):
        params = _suggest_params(trial, parent_lr)
        lr = float(params["lr"])
        model, summary = train(
            recipe,
            proxy_ids,
            lr,
            settings.PROXY_STEPS,
            settings.TRIAL_SEED_BASE + trial.number,
            cache_dir,
            patch_size=settings.PATCH_SIZE,
            batch_size=settings.BATCH_SIZE,
            grad_clip_norm=settings.GRAD_CLIP_NORM,
            nan_abort_steps=settings.NAN_ABORT_STEPS,
            augment_timeout_s=settings.AUGMENT_TIMEOUT_S,
        )
        if model is None:
            return 0.0
        score, _, reason = _score_cases(model, optuna_ids, cache_dir)
        if reason:
            trial.set_user_attr("reason", reason)
            return 0.0
        return score

    study.optimize(objective, n_trials=settings.OPTUNA_N_TRIALS)
    best_lr = float(study.best_params.get("lr", settings.DEFAULT_LR))

    model, summary = train(
        recipe,
        proxy_ids,
        best_lr,
        settings.PROXY_STEPS,
        settings.REEVAL_SEED,
        cache_dir,
        patch_size=settings.PATCH_SIZE,
        batch_size=settings.BATCH_SIZE,
        grad_clip_norm=settings.GRAD_CLIP_NORM,
        nan_abort_steps=settings.NAN_ABORT_STEPS,
        augment_timeout_s=settings.AUGMENT_TIMEOUT_S,
    )
    if model is None:
        return _fail(summary.get("reason", "reeval_failed"), stage1_ok=1.0)

    fitness, per_class, reason = _score_cases(model, fitness_ids, cache_dir)
    if reason:
        return _fail(reason, stage1_ok=1.0, best_lr=best_lr)

    n_params = stage1["n_params"]
    metrics = {
        "ok": True,
        "stage1_ok": 1.0,
        "combined_score": float(fitness),
        "worst_class_dice": float(min(per_class) if per_class else 0.0),
        "params_millions": float(n_params / 1e6),
        "best_lr": float(best_lr),
        "train_curve": summary.get("losses", []),
    }
    for i, d in enumerate(per_class, start=1):
        metrics[f"dice_class_{i}"] = float(d)
    return metrics


def run_stage(program_path: str, stage: str) -> dict:
    if stage == "stage1":
        return run_stage1(program_path)
    if stage == "stage2":
        return run_stage2(program_path)
    raise ValueError(f"unknown stage {stage}")
