"""OpenEvolve evaluator contract. Thin wrapper around isolated stage workers."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

TRACK_B = Path(__file__).resolve().parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()

import settings  # noqa: E402
from candidate_runner import run_isolated  # noqa: E402
from harness import guards  # noqa: E402

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # pragma: no cover - allows unit tests without the package
    class EvaluationResult:  # type: ignore
        def __init__(self, metrics, artifacts=None):
            self.metrics = metrics
            self.artifacts = artifacts or {}


def _stamp_manifest() -> None:
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.OUTPUT_DIR / "run_manifest.json"
    if path.exists():
        return
    import hashlib
    import subprocess

    def _git(cmd):
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return "unknown"

    patch = TRACK_B / "patches" / "parent_lr.py"
    payload = {
        "settings": settings.as_dict(),
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["git", "status", "--porcelain"])),
        "openevolve_version": _git(["python", "-c", "import openevolve,sys; print(openevolve.__version__)"]),
        "parent_lr_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest() if patch.exists() else None,
        "frozen_hashes": json.loads((TRACK_B / "harness" / "frozen_hashes.json").read_text())
        if (TRACK_B / "harness" / "frozen_hashes.json").exists()
        else {},
        "profile": settings.PROFILE,
        "created_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2))


def _result(metrics: dict, artifacts: dict | None = None) -> EvaluationResult:
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            clean[k] = float(v)
    return EvaluationResult(metrics=clean, artifacts=artifacts or {})


def evaluate_stage1(program_path: str):
    try:
        _stamp_manifest()
        payload = run_isolated(program_path, "stage1", timeout=min(120, settings.EVALUATOR_TIMEOUT))
        if not payload.get("ok"):
            return _result(
                {"stage1_ok": 0.0, "combined_score": 0.0},
                {"stderr": payload.get("reason") or payload.get("stderr") or "stage1 failed"},
            )
        return _result(
            {
                "stage1_ok": 1.0,
                "combined_score": 1.0,
                "params_millions": float(payload.get("params_millions", 0.0)),
            },
            {"stderr": ""},
        )
    except Exception as exc:
        return _result(
            {"stage1_ok": 0.0, "combined_score": 0.0},
            {"stderr": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]},
        )


def evaluate_stage2(program_path: str):
    try:
        extra = {}
        sidecar = program_path + ".parent.json"
        if Path(sidecar).exists():
            extra["FETA_PARENT_SIDECAR"] = sidecar
        payload = run_isolated(
            program_path,
            "stage2",
            extra_env=extra,
            timeout=settings.EVALUATOR_TIMEOUT,
        )
        if not payload.get("ok"):
            return _result(
                {
                    "stage1_ok": float(payload.get("stage1_ok", 0.0)),
                    "combined_score": 0.0,
                    "worst_class_dice": 0.0,
                    "params_millions": float(payload.get("params_millions", 0.0)),
                    "best_lr": float(payload.get("best_lr", settings.DEFAULT_LR)),
                },
                {"stderr": payload.get("reason") or payload.get("stderr") or "stage2 failed"},
            )
        metrics = {
            "combined_score": float(payload["combined_score"]),
            "worst_class_dice": float(payload["worst_class_dice"]),
            "params_millions": float(payload["params_millions"]),
            "best_lr": float(payload["best_lr"]),
            "stage1_ok": 1.0,
        }
        for i in range(1, 8):
            key = f"dice_class_{i}"
            if key in payload:
                metrics[key] = float(payload[key])
        artifacts = {
            "stderr": "",
            "train_curve": json.dumps(payload.get("train_curve", []))[:8000],
        }
        return _result(metrics, artifacts)
    except Exception as exc:
        return _result(
            {"combined_score": 0.0, "stage1_ok": 0.0},
            {"stderr": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]},
        )


def evaluate(program_path: str):
    """Full evaluation used if cascade is disabled."""
    s1 = evaluate_stage1(program_path)
    metrics = getattr(s1, "metrics", s1)
    if float(metrics.get("stage1_ok", 0.0)) < 1.0:
        return s1
    return evaluate_stage2(program_path)
