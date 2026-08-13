from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from harness import guards
from prompts.lint_system_message import lint


MALICIOUS = {
    "nnunet_import": '''
import torch
import nnunetv2
def build_model(in_channels, num_classes):
    return torch.nn.Identity()
def build_loss():
    return lambda o, t: t.sum() * 0
def build_optimizer(params, lr):
    return torch.optim.SGD(params, lr)
def build_scheduler(optimizer, total_steps):
    return None
def build_sampler():
    return lambda l, p, r: (0, 0, 0)
def build_augmentation():
    return lambda i, l, r: (i, l)
''',
    "constant": None,  # filled from seed with constant forward
    "bare_except": None,
    "split_edit": None,
    "fabricate": None,
}


def _wrap_evolve(body: str, track_b: Path) -> str:
    seed = (track_b / "initial_program.py").read_text()
    pre = seed.split("# EVOLVE-BLOCK-START", 1)[0] + "# EVOLVE-BLOCK-START\n"
    post = "\n# EVOLVE-BLOCK-END" + seed.split("# EVOLVE-BLOCK-END", 1)[1]
    return pre + body + post


def test_forbidden_nnunet_import(tmp_path, track_b):
    path = tmp_path / "bad.py"
    path.write_text(_wrap_evolve("import nnunetv2\n" + MALICIOUS["nnunet_import"].split("import nnunetv2", 1)[1], track_b))
    with pytest.raises(ValueError, match="forbidden"):
        guards.check_imports(path)


def test_bare_except_rejected(tmp_path, track_b):
    body = """
import torch
import torch.nn as nn
import numpy as np
def build_model(in_channels, num_classes):
    try:
        return nn.Identity()
    except:
        return nn.Identity()
def build_loss():
    return lambda o, t: t.float().mean() * 0
def build_optimizer(params, lr):
    return torch.optim.Adam(params, lr)
def build_scheduler(optimizer, total_steps):
    return None
def build_sampler():
    return lambda l, p, r: (0, 0, 0)
def build_augmentation():
    return lambda i, l, r: (i, l)
"""
    path = tmp_path / "bare.py"
    path.write_text(_wrap_evolve(body, track_b))
    with pytest.raises(ValueError, match="bare except"):
        guards.check_bare_except(path)


def test_outside_block_edit_rejected(tmp_path, track_b):
    seed = (track_b / "initial_program.py").read_text()
    path = tmp_path / "edited.py"
    path.write_text(seed.replace("NUM_CLASSES = 8", "NUM_CLASSES = 9", 1))
    with pytest.raises(ValueError, match="outside the evolve block"):
        guards.frozen_prefix_suffix(track_b / "initial_program.py", path)


def test_constant_prediction_insane():
    pred = np.ones((32, 32, 32), dtype=np.int64)
    with pytest.raises(AssertionError):
        guards.assert_prediction_sane(pred, reference_shape=(32, 32, 32))


def test_split_integrity_rejects_overlap():
    splits = {"train": ["a"], "tuning": ["a"], "test": ["b"]}
    subs = {
        "proxy_train": ["a"],
        "optuna_selection": ["a"],
        "fitness": [],
        "sealed_test": ["b"],
    }
    with pytest.raises(AssertionError):
        guards.assert_split_integrity(subs, splits)


def test_system_message_no_leakage(track_b):
    text = (track_b / "prompts" / "system_message.txt").read_text()
    assert lint(text) == []
