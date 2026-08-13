"""Figures 1–3 and Tables 1–3 from results JSON. Never types numbers by hand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRACK_B = Path(__file__).resolve().parent.parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()

OUT = REPO_ROOT / "results" / "figures"


def _load(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def figure1(target_trace: dict, random_trace: dict, zero_shot_hits: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if target_trace:
        ax.step(target_trace["gpu_hours"], target_trace["cumulative"], where="post", label="evolution")
    if random_trace:
        ax.step(random_trace["gpu_hours"], random_trace["cumulative"], where="post", label="random mutation")
    ax.axhline(zero_shot_hits, linestyle="--", label="zero-shot")
    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Targets recovered (0–8)")
    ax.set_ylim(0, 8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure1_targets.png", dpi=150)
    plt.close(fig)


def figure2(fitness_trace: dict, seed_full: float | None, nnunet: float | None, sigma: float | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if fitness_trace:
        x = np.asarray(fitness_trace["gpu_hours"])
        y = np.asarray(fitness_trace["best"])
        ax.plot(x, y, label="best so far")
        if sigma is not None:
            ax.fill_between(x, y - 2 * sigma, y + 2 * sigma, alpha=0.2, label="±2σ noise")
    if seed_full is not None:
        ax.axhline(seed_full, linestyle=":", label="seed full training")
    if nnunet is not None:
        ax.axhline(nnunet, linestyle="--", label="stock nnU-Net")
    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Mean Dice (classes 1–7)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure2_fitness.png", dpi=150)
    plt.close(fig)


def figure3(archive: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if archive:
        x = [c["params_millions"] for c in archive]
        y = [c["worst_class_dice"] for c in archive]
        c = [c["mean_dice"] for c in archive]
        sc = ax.scatter(x, y, c=c, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="mean Dice")
    ax.set_xlabel("params_millions")
    ax.set_ylabel("worst_class_dice")
    fig.tight_layout()
    fig.savefig(out / "figure3_archive.png", dpi=150)
    plt.close(fig)


def tables(target_scores: dict, sealed: dict | None, offlist: list, out: Path) -> None:
    table1 = []
    recovered = (target_scores or {}).get("recovered_on_elites", {})
    first = (target_scores or {}).get("first_appearance", {})
    for t in [f"T{i}" for i in range(1, 9)]:
        table1.append(
            {
                "target": t,
                "recovered": bool(recovered.get(t)),
                "first_iteration": (first.get(t) or {}).get("iteration"),
                "gpu_hours_first": (first.get(t) or {}).get("gpu_hours"),
            }
        )
    (out / "table1_targets.json").write_text(json.dumps(table1, indent=2))
    (out / "table2_sealed.json").write_text(json.dumps(sealed or {"status": "sealed"}, indent=2))
    (out / "table3_offlist.json").write_text(json.dumps(offlist or [], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    out = args.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    target_scores = _load(args.results_dir / "target_scores.json", {})
    calibration = _load(args.results_dir / "calibration.json", {})
    traces = _load(args.results_dir / "evolution_traces.json", {})
    zero = _load(args.results_dir / "zero_shot_targets.json", [])
    zero_hits = 0.0
    if zero:
        zero_hits = float(np.mean([item.get("n_hit", 0) for item in zero]))
    figure1(traces.get("evolution_targets"), traces.get("random_targets"), zero_hits, out)
    figure2(
        traces.get("fitness"),
        traces.get("seed_full_dice"),
        traces.get("nnunet_dice"),
        (calibration.get("noise_floor") or {}).get("std"),
        out,
    )
    figure3(traces.get("archive") or [], out)
    tables(target_scores, _load(args.results_dir / "sealed_test_scores.json"), traces.get("offlist") or [], out)
    print(f"Wrote figures under {out}")


if __name__ == "__main__":
    main()
