"""Noise-floor and proxy rank-correlation calibration.

The ρ < 0.6 gate stops the experiment. Escalation to 8000 steps / 128³ is a
pre-registered protocol change and is not applied automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TRACK_B = Path(__file__).resolve().parent.parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()

import settings  # noqa: E402
from harness.guards import load_recipe  # noqa: E402
from harness.train_loop import train  # noqa: E402
from evaluator_impl import _load_splits, _score_cases  # noqa: E402

VARIANT_NAMES = [
    "seed",
    "instance_norm",
    "dice_term",
    "deep_supervision",
    "lr_10x",
    "base8",
    "sgd_poly",
    "foreground_sampling",
]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return float(rho)


def noise_floor(n: int = 5) -> dict:
    splits, subs = _load_splits()
    recipe = load_recipe(TRACK_B / "initial_program.py")
    scores = []
    for i in range(n):
        model, summary = train(
            recipe,
            subs["proxy_train"],
            settings.DEFAULT_LR,
            settings.PROXY_STEPS,
            seed=100 + i,
            cache_dir=settings.CACHE_DIR,
            patch_size=settings.PATCH_SIZE,
            batch_size=settings.BATCH_SIZE,
        )
        if model is None:
            scores.append(0.0)
            continue
        fitness, _, _ = _score_cases(model, subs["fitness"], settings.CACHE_DIR)
        scores.append(fitness)
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "scores": [float(x) for x in arr],
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "n": n,
    }


class _RecipeProxy:
    def __init__(self, base, **overrides):
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def _variant_recipe(name: str):
    """Hand-built variants spanning good to bad. Registered here, not discovered post-hoc."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    base = load_recipe(TRACK_B / "initial_program.py")
    if name == "seed":
        return base
    if name == "lr_10x":
        return base
    if name == "instance_norm":
        orig = base.build_model

        def build_model(in_channels, num_classes):
            model = orig(in_channels, num_classes)
            for m in list(model.modules()):
                pass
            def _swap(module):
                for child_name, child in list(module.named_children()):
                    if isinstance(child, nn.BatchNorm3d):
                        setattr(module, child_name, nn.InstanceNorm3d(child.num_features, affine=True))
                    else:
                        _swap(child)
            _swap(model)
            return model

        return _RecipeProxy(base, build_model=build_model)
    if name == "dice_term":
        orig_loss = base.build_loss

        def build_loss():
            ce_fn = orig_loss()

            def loss_fn(output, target):
                logits = output[0] if isinstance(output, (list, tuple)) else output
                ce = ce_fn(output, target)
                prob = torch.softmax(logits, dim=1)
                dice = 0.0
                for c in range(1, 8):
                    p = prob[:, c]
                    g = (target == c).float()
                    dice = dice + (1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1))
                return ce + dice / 7

            return loss_fn

        return _RecipeProxy(base, build_loss=build_loss)
    if name == "deep_supervision":
        orig = base.build_model

        def build_model(in_channels, num_classes):
            model = orig(in_channels, num_classes)
            orig_fwd = model.forward

            def forward(x):
                y = orig_fwd(x)
                return (y, y)

            model.forward = forward
            return model

        orig_loss = base.build_loss

        def build_loss():
            inner = orig_loss()

            def loss_fn(output, target):
                return inner(output[0], target) + 0.5 * inner(output[1], target)

            return loss_fn

        return _RecipeProxy(base, build_model=build_model, build_loss=build_loss)
    if name == "base8":
        from initial_program import UNet3D

        def build_model(in_channels, num_classes):
            return UNet3D(in_channels=in_channels, num_classes=num_classes, base=8)

        return _RecipeProxy(base, build_model=build_model)
    if name == "sgd_poly":
        def build_optimizer(params, lr):
            return torch.optim.SGD(params, lr=lr, momentum=0.99, nesterov=True)

        def build_scheduler(optimizer, total_steps):
            def lr_lambda(step):
                return (1 - min(step, total_steps) / max(total_steps, 1)) ** 0.9

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return _RecipeProxy(base, build_optimizer=build_optimizer, build_scheduler=build_scheduler)
    if name == "foreground_sampling":
        import numpy as np

        def build_sampler():
            def sample_origin(label_volume, patch_size, rng):
                if rng.random() < 1 / 3:
                    fg = np.argwhere(label_volume > 0)
                    if len(fg):
                        centre = fg[int(rng.integers(0, len(fg)))]
                        origin = []
                        for i in range(3):
                            o = int(centre[i] - patch_size[i] // 2)
                            o = max(0, min(label_volume.shape[i] - patch_size[i], o))
                            origin.append(o)
                        return tuple(origin)
                return tuple(
                    int(rng.integers(0, label_volume.shape[i] - patch_size[i] + 1))
                    for i in range(3)
                )

            return sample_origin

        return _RecipeProxy(base, build_sampler=build_sampler)
    raise ValueError(name)


def rank_gate(full_steps: int | None = None) -> dict:
    splits, subs = _load_splits()
    proxy_scores = []
    # Full-training scores are supplied later on A6000; here we record proxy only
    # unless --full is passed.
    for name in VARIANT_NAMES:
        recipe = _variant_recipe(name)
        lr = settings.DEFAULT_LR * (10 if name == "lr_10x" else 1)
        model, _ = train(
            recipe,
            subs["proxy_train"],
            lr,
            settings.PROXY_STEPS,
            seed=0,
            cache_dir=settings.CACHE_DIR,
            patch_size=settings.PATCH_SIZE,
            batch_size=settings.BATCH_SIZE,
        )
        score = 0.0 if model is None else _score_cases(model, subs["fitness"], settings.CACHE_DIR)[0]
        proxy_scores.append({"name": name, "proxy": score})
    return {"proxy": proxy_scores, "variants": VARIANT_NAMES}


def decide(rho: float) -> str:
    if rho >= 0.7:
        return "proceed"
    if rho >= 0.6:
        return "proceed_with_limitation"
    return "stop_escalate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "calibration.json")
    parser.add_argument("--full-scores", type=Path, default=None, help="JSON list of full-training scores aligned to VARIANT_NAMES")
    args = parser.parse_args()
    payload = {"profile": settings.PROFILE}
    payload["noise_floor"] = noise_floor(5)
    if not args.noise_only:
        payload["rank_gate"] = rank_gate()
        if args.full_scores:
            full = json.loads(Path(args.full_scores).read_text())
            proxy = np.array([x["proxy"] for x in payload["rank_gate"]["proxy"]], dtype=np.float64)
            full_arr = np.array(full, dtype=np.float64)
            rho = _spearman(proxy, full_arr)
            payload["spearman_rho"] = rho
            payload["decision"] = decide(rho)
            if payload["decision"] == "stop_escalate":
                payload["escalation"] = {
                    "steps": 8000,
                    "patch": [128, 128, 128],
                    "iterations": 80,
                    "note": "Do not apply automatically. Requires a dated PREREGISTRATION amendment.",
                }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
