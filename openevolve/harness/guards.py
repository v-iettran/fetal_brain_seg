"""Anti-reward-hacking assertions. Frozen."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from harness.constants import (
    ALLOWED_IMPORT_ROOTS,
    CONTRACT_FUNCTIONS,
    EVOLVE_END,
    EVOLVE_START,
    FORBIDDEN_IMPORT_SUBSTRINGS,
    NUM_CLASSES,
)

ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = Path(__file__).resolve().parent
OPENEVOLVE_DIR = HARNESS_DIR.parent
HASH_MANIFEST = OPENEVOLVE_DIR / "harness" / "frozen_hashes.json"

FROZEN_RELATIVE = [
    "openevolve/harness/__init__.py",
    "openevolve/harness/constants.py",
    "openevolve/harness/prepare.py",
    "openevolve/harness/data.py",
    "openevolve/harness/device.py",
    "openevolve/harness/train_loop.py",
    "openevolve/harness/infer.py",
    "openevolve/harness/guards.py",
    "openevolve/evaluator.py",
    "src/metrics.py",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen_hashes() -> dict:
    payload = {rel: sha256_file(ROOT / rel) for rel in FROZEN_RELATIVE}
    HASH_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def verify_frozen_files() -> None:
    if not HASH_MANIFEST.exists():
        raise RuntimeError(f"missing frozen hash manifest {HASH_MANIFEST}")
    expected = json.loads(HASH_MANIFEST.read_text())
    mismatches = []
    for rel, digest in expected.items():
        path = ROOT / rel
        if not path.exists():
            mismatches.append(f"missing {rel}")
            continue
        got = sha256_file(path)
        if got != digest:
            mismatches.append(f"{rel}: expected {digest[:12]} got {got[:12]}")
    if mismatches:
        raise RuntimeError("frozen file hash mismatch: " + "; ".join(mismatches))


def _module_root(name: str) -> str:
    return name.split(".")[0]


def check_imports(program_path) -> None:
    source = Path(program_path).read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    for name in imported:
        lowered = name.lower()
        for bad in FORBIDDEN_IMPORT_SUBSTRINGS:
            if bad.lower() in lowered:
                raise ValueError(f"forbidden import {name}")
        root = _module_root(name)
        if root not in ALLOWED_IMPORT_ROOTS and name not in ALLOWED_IMPORT_ROOTS:
            raise ValueError(f"import {name} is not on the whitelist")


def _evolve_source(source: str) -> str:
    if EVOLVE_START not in source or EVOLVE_END not in source:
        raise ValueError("program is missing EVOLVE-BLOCK markers")
    start = source.index(EVOLVE_START) + len(EVOLVE_START)
    end = source.index(EVOLVE_END)
    if end <= start:
        raise ValueError("invalid EVOLVE-BLOCK markers")
    return source[start:end]


def check_bare_except(program_path) -> None:
    block = _evolve_source(Path(program_path).read_text())
    tree = ast.parse(block)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            raise ValueError("bare except is forbidden inside the evolve block")


def check_forbidden_calls(program_path) -> None:
    source = Path(program_path).read_text()
    tree = ast.parse(source)
    forbidden = {"system", "popen", "Popen", "urlopen", "load_state_dict"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen"}:
            raise ValueError(f"forbidden call os.{node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in forbidden:
                raise ValueError(f"forbidden call {name}")


def load_recipe(program_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_program"] = module
    spec.loader.exec_module(module)
    return module


def check_contract(recipe) -> None:
    for name in CONTRACT_FUNCTIONS:
        if not hasattr(recipe, name):
            raise ValueError(f"missing contract function {name}")
        fn = getattr(recipe, name)
        if not callable(fn):
            raise ValueError(f"{name} is not callable")
        n_req = sum(
            p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in inspect.signature(fn).parameters.values()
        )
        expected = {
            "build_model": 2,
            "build_loss": 0,
            "build_optimizer": 2,
            "build_scheduler": 2,
            "build_sampler": 0,
            "build_augmentation": 0,
        }[name]
        if n_req != expected:
            raise ValueError(f"{name} has {n_req} required args, expected {expected}")


def frozen_prefix_suffix(seed_path: Path, candidate_path: Path) -> None:
    seed = seed_path.read_text()
    cand = candidate_path.read_text()
    if EVOLVE_START not in cand or EVOLVE_END not in cand:
        raise ValueError("candidate removed EVOLVE-BLOCK markers")
    seed_pre = seed.split(EVOLVE_START, 1)[0]
    seed_post = seed.split(EVOLVE_END, 1)[1]
    cand_pre = cand.split(EVOLVE_START, 1)[0]
    cand_post = cand.split(EVOLVE_END, 1)[1]
    if seed_pre != cand_pre or seed_post != cand_post:
        raise ValueError("code outside the evolve block was modified")


def assert_split_integrity(subsplits: dict, splits: dict) -> None:
    train = set(splits["train"])
    tune = set(splits["tuning"])
    test = set(splits["test"])
    if not train.isdisjoint(tune) or not train.isdisjoint(test) or not tune.isdisjoint(test):
        raise AssertionError("master splits are not disjoint")
    proxy = set(subsplits["proxy_train"])
    optuna = set(subsplits["optuna_selection"])
    fitness = set(subsplits["fitness"])
    sealed = set(subsplits["sealed_test"])
    if proxy != train:
        raise AssertionError("proxy_train must equal splits.train")
    if optuna | fitness != tune:
        raise AssertionError("optuna+fitness must equal splits.tuning")
    if not optuna.isdisjoint(fitness):
        raise AssertionError("optuna and fitness overlap")
    if sealed != test:
        raise AssertionError("sealed_test must equal splits.test")
    if not proxy.isdisjoint(optuna | fitness | sealed):
        raise AssertionError("training cases leak into held-out sets")


def assert_prediction_sane(pred: np.ndarray, reference_shape=None) -> None:
    if pred.dtype != np.int64:
        raise AssertionError(f"prediction dtype {pred.dtype} != int64")
    if reference_shape is not None and tuple(pred.shape) != tuple(reference_shape):
        raise AssertionError(f"prediction shape {pred.shape} != {reference_shape}")
    uniq = set(int(v) for v in np.unique(pred))
    if len(uniq) < 5:
        raise AssertionError(f"degenerate prediction: only {sorted(uniq)}")
    fg = pred > 0
    if not fg.any():
        raise AssertionError("prediction has no foreground")
    for c in uniq:
        if c == 0:
            continue
        frac = float((pred == c).sum()) / float(fg.sum())
        if frac > 0.90:
            raise AssertionError(f"class {c} occupies {frac:.2%} of foreground")
        if frac == 0.0:
            raise AssertionError(f"class {c} occupies 0% of foreground")
    if not uniq <= set(range(NUM_CLASSES)):
        raise AssertionError(f"unexpected labels {sorted(uniq)}")
