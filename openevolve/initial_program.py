"""
Seed program — Track B, OpenEvolve.

3D U-Net (Cicek et al., MICCAI 2016) with a deliberately minimal training recipe.
See PREREGISTRATION.md for the documented deviations from the paper's recipe.

CONTRACT (enforced by the harness — do not change these six signatures):
    build_model(in_channels, num_classes) -> nn.Module
    build_loss()                          -> (output, target) -> scalar
    build_optimizer(params, lr)           -> Optimizer
    build_scheduler(optimizer, total_steps) -> scheduler or None
    build_sampler()                       -> (label_vol, patch_size, rng) -> origin
    build_augmentation()                  -> (img, lbl, rng) -> (img, lbl)
"""
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

NUM_CLASSES = 8
IN_CHANNELS = 1

# EVOLVE-BLOCK-START


def _block(c_in: int, c_mid: int, c_out: int) -> nn.Sequential:
    """Two 3x3x3 convolutions, each followed by normalisation and activation."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_mid, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(c_mid),
        nn.ReLU(inplace=True),
        nn.Conv3d(c_mid, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 8, base: int = 32):
        super().__init__()
        b = base
        self.enc0 = _block(in_channels, b, 2 * b)      # 1  -> 32  -> 64
        self.enc1 = _block(2 * b, 2 * b, 4 * b)        # 64 -> 64  -> 128
        self.enc2 = _block(4 * b, 4 * b, 8 * b)        # 128-> 128 -> 256
        self.bottom = _block(8 * b, 8 * b, 16 * b)     # 256-> 256 -> 512

        self.pool = nn.MaxPool3d(2)

        self.up2 = nn.ConvTranspose3d(16 * b, 16 * b, kernel_size=2, stride=2)
        self.dec2 = _block(16 * b + 8 * b, 8 * b, 8 * b)    # 768 -> 256 -> 256
        self.up1 = nn.ConvTranspose3d(8 * b, 8 * b, kernel_size=2, stride=2)
        self.dec1 = _block(8 * b + 4 * b, 4 * b, 4 * b)     # 384 -> 128 -> 128
        self.up0 = nn.ConvTranspose3d(4 * b, 4 * b, kernel_size=2, stride=2)
        self.dec0 = _block(4 * b + 2 * b, 2 * b, 2 * b)     # 192 -> 64  -> 64

        self.out = nn.Conv3d(2 * b, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(self.pool(s1))
        x = self.bottom(self.pool(s2))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        x = self.dec0(torch.cat([self.up0(x), s0], dim=1))
        return self.out(x)


def build_model(in_channels: int, num_classes: int) -> nn.Module:
    return UNet3D(in_channels=in_channels, num_classes=num_classes, base=32)


def build_loss() -> Callable[[Any, torch.Tensor], torch.Tensor]:
    ce = nn.CrossEntropyLoss()

    def loss_fn(output: Any, target: torch.Tensor) -> torch.Tensor:
        logits = output[0] if isinstance(output, (list, tuple)) else output
        return ce(logits, target)

    return loss_fn


def build_optimizer(params: Iterable, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr)


def build_scheduler(optimizer, total_steps: int) -> Optional[Any]:
    return None  # constant learning rate


def build_sampler() -> Callable:
    def sample_origin(label_volume, patch_size, rng):
        return tuple(
            int(rng.integers(0, label_volume.shape[i] - patch_size[i] + 1))
            for i in range(3)
        )
    return sample_origin


def build_augmentation() -> Callable:
    def augment(image, label, rng):
        for axis in range(3):
            if rng.random() < 0.5:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)
        k = int(rng.integers(0, 4))
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(1, 2))
        return np.ascontiguousarray(image), np.ascontiguousarray(label)
    return augment


# EVOLVE-BLOCK-END
