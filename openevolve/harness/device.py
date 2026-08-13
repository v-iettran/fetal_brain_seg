"""Device selection. Production is CUDA; smoke may use MPS or CPU."""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    forced = os.environ.get("FETA_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("FETA_PROFILE") == "smoke" and torch.backends.mps.is_available():
        return torch.device("mps")
    if os.environ.get("FETA_ALLOW_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        # Production profile on a Mac: refuse to silently run a scientific job.
        profile = os.environ.get("FETA_PROFILE", "production")
        if profile == "smoke":
            return torch.device("mps")
        raise RuntimeError(
            "CUDA is required for the production profile. Set FETA_PROFILE=smoke "
            "for local MPS debugging, which is not scientific."
        )
    raise RuntimeError("No CUDA device. Set FETA_PROFILE=smoke and FETA_ALLOW_CPU=1 for CPU debug.")


def amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def autocast_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.float16
    return torch.float32
