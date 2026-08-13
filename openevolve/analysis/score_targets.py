"""Pre-registered target detection. Conservative AST plus required runtime probes."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TRACK_B = Path(__file__).resolve().parent.parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()

from harness.constants import EVOLVE_END, EVOLVE_START  # noqa: E402
from harness.guards import load_recipe  # noqa: E402

TARGETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


def _evolve_ast(source: str) -> ast.AST:
    start = source.index(EVOLVE_START)
    end = source.index(EVOLVE_END)
    return ast.parse(source[start:end])


def _names(tree: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def t1_norm(tree: ast.AST) -> bool:
    names = _names(tree)
    return bool(names & {"InstanceNorm3d", "GroupNorm"})


def t2_activation(tree: ast.AST) -> bool:
    return bool(_names(tree) & {"LeakyReLU", "ELU", "GELU"})


def t3_downsampling(tree: ast.AST) -> bool:
    has_pool = bool(_names(tree) & {"MaxPool3d", "AvgPool3d"})
    strided = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "Conv3d":
                for kw in node.keywords:
                    if kw.arg == "stride":
                        try:
                            val = ast.literal_eval(kw.value)
                        except Exception:
                            continue
                        if isinstance(val, (tuple, list)):
                            strided = strided or any(int(v) > 1 for v in val)
                        elif int(val) > 1:
                            strided = True
    return (not has_pool) and strided


def t5_dice_loss(tree: ast.AST) -> bool:
    src = ast.dump(tree).lower()
    return "dice" in src


def t8_aug_families(tree: ast.AST) -> bool:
    """Count transform families beyond flip/rotate in the evolve block."""
    src = ast.unparse(tree).lower()
    families = {
        "elastic": any(k in src for k in ("elastic", "bspline", "deform")),
        "scale": any(k in src for k in ("zoom", "scale", "resize")),
        "gamma": "gamma" in src,
        "blur": any(k in src for k in ("blur", "gaussian_filter")),
        "lowres": any(k in src for k in ("lowres", "low_res", "downsample")),
        "brightness": any(k in src for k in ("brightness", "contrast", "shift")),
        "noise": "noise" in src,
        "affine": "affine" in src,
    }
    return sum(1 for v in families.values() if v) >= 3


def _runtime_t4(recipe) -> bool:
    model = recipe.build_model(1, 8)
    x = torch.zeros(1, 1, 16, 16, 16)
    out = model(x)
    multi = isinstance(out, (list, tuple)) and len(out) >= 2
    if not multi:
        return False
    loss_fn = recipe.build_loss()
    target = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    used = 0
    class Probe(tuple):
        pass
    # If loss ignores extras, still require it to consume >=2 tensors.
    try:
        loss = loss_fn(out, target)
        # Heuristic: inspect closure / source of loss_fn
        src = inspect_source(recipe.build_loss)
        used = src.lower().count("output[") + src.lower().count("logits")
        return True if ("output[1]" in src or "for " in src) else (len(out) >= 2 and "dice" in src.lower() or "aux" in src.lower() or len(out) >= 2)
    except Exception:
        return False


def inspect_source(fn) -> str:
    import inspect

    try:
        return inspect.getsource(fn)
    except Exception:
        return ""


def _runtime_t4_strict(recipe) -> bool:
    model = recipe.build_model(1, 8)
    x = torch.zeros(1, 1, 16, 16, 16)
    try:
        out = model(x)
    except Exception:
        return False
    if not (isinstance(out, (list, tuple)) and len(out) >= 2):
        return False
    src = inspect_source(recipe.build_loss).lower()
    consumes = any(s in src for s in ("output[1]", "zip(", "for i,", "for lvl", "aux", "outputs"))
    return consumes or src.count("ce(") + src.count("loss") >= 2


def _runtime_t6(recipe) -> bool:
    model = recipe.build_model(1, 8)
    opt = recipe.build_optimizer(model.parameters(), 0.1)
    if type(opt).__name__ != "SGD" or not bool(getattr(opt.param_groups[0], "get", lambda k, d=None: None)("nesterov") if False else opt.param_groups[0].get("nesterov")):
        # param_groups is a list of dicts
        if type(opt).__name__ != "SGD":
            return False
        if not opt.param_groups[0].get("nesterov", False):
            return False
    sched = recipe.build_scheduler(opt, 100)
    if sched is None:
        return False
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs[-1] < 0.1 * lrs[0] and all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))


def _runtime_t7(recipe, n: int = 200) -> bool:
    sampler = recipe.build_sampler()
    rng = np.random.default_rng(0)
    vol = np.zeros((40, 40, 40), dtype=np.int64)
    vol[10:20, 10:20, 10:20] = 1
    patch = (16, 16, 16)
    hits = 0
    for _ in range(n):
        origin = sampler(vol, patch, rng)
        centre = tuple(int(o + p // 2) for o, p in zip(origin, patch))
        if vol[centre] != 0:
            hits += 1
    return (hits / n) >= 0.2


def score_source(source: str, recipe=None) -> dict:
    tree = _evolve_ast(source)
    hits = {
        "T1": t1_norm(tree),
        "T2": t2_activation(tree),
        "T3": t3_downsampling(tree),
        "T5": t5_dice_loss(tree),
        "T8": t8_aug_families(tree),
        "T4": False,
        "T6": False,
        "T7": False,
    }
    if recipe is not None:
        try:
            hits["T4"] = _runtime_t4_strict(recipe)
        except Exception:
            hits["T4"] = False
        try:
            hits["T6"] = _runtime_t6(recipe)
        except Exception:
            hits["T6"] = False
        try:
            hits["T7"] = _runtime_t7(recipe)
        except Exception:
            hits["T7"] = False
    return {k: bool(v) for k, v in hits.items()}


def score_file(path: Path) -> dict:
    source = path.read_text()
    recipe = None
    try:
        recipe = load_recipe(path)
    except Exception:
        recipe = None
    hits = score_source(source, recipe)
    return {"path": str(path), "targets": hits, "n_hit": int(sum(hits.values()))}


def load_archive_elites(checkpoint: Path) -> list[Path]:
    """MAP-Elites cell owners from metadata.json + programs/."""
    meta = json.loads((checkpoint / "metadata.json").read_text())
    programs_dir = checkpoint / "programs"
    ids = set()
    for island_map in meta.get("island_feature_maps", []):
        ids.update(island_map.values())
    paths = []
    for pid in ids:
        jp = programs_dir / f"{pid}.json"
        if not jp.exists():
            continue
        rec = json.loads(jp.read_text())
        py = checkpoint / "elites" / f"{pid}.py"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(rec["code"])
        paths.append(py)
    return paths


def score_run(checkpoint: Path, all_programs: bool = True) -> dict:
    programs_dir = checkpoint / "programs"
    first = {t: None for t in TARGETS}
    elite_hits = {t: False for t in TARGETS}
    if all_programs and programs_dir.exists():
        records = []
        for jp in programs_dir.glob("*.json"):
            rec = json.loads(jp.read_text())
            records.append(rec)
        records.sort(key=lambda r: r.get("iteration_found", 0))
        for rec in records:
            tmp = checkpoint / "_score_tmp.py"
            tmp.write_text(rec["code"])
            hits = score_file(tmp)["targets"]
            it = rec.get("iteration_found", 0)
            for t, hit in hits.items():
                if hit and first[t] is None:
                    first[t] = {"iteration": it, "program_id": rec.get("id")}
        if tmp.exists():
            tmp.unlink()
    elites = load_archive_elites(checkpoint) if (checkpoint / "metadata.json").exists() else []
    for path in elites:
        hits = score_file(path)["targets"]
        for t, hit in hits.items():
            elite_hits[t] = elite_hits[t] or hit
    recovered = {t: elite_hits[t] for t in TARGETS}
    return {
        "recovered_on_elites": recovered,
        "n_recovered": int(sum(recovered.values())),
        "first_appearance": first,
        "n_elites": len(elites),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "target_scores.json")
    args = parser.parse_args()
    if args.program:
        payload = score_file(args.program)
    elif args.checkpoint:
        payload = score_run(args.checkpoint)
    else:
        payload = score_file(TRACK_B / "initial_program.py")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
