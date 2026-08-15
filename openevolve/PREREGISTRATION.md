# Track B pre-registration — OpenEvolve on FeTA

**Status:** immutable after the pre-registration commit. Later changes belong in a
dated amendment section at the bottom of this file.

**Owners:** Viet, James
**Hardware (production):** 1 × RTX A6000 (48 GB), CUDA
**OpenEvolve pin:** v0.3.2 (`411fb59c886c18704caaffb611e17cf9e7d824d2`)
**Random seed:** 42

This document is the answer key for the experiment. The LLM system message must
not name or unambiguously paraphrase any target below.

---

## Fidelity statement

The **architecture** is faithful to Çiçek et al., MICCAI 2016. The **training
recipe** is deliberately minimal, because the paper's own recipe would
pre-satisfy two targets. Every deviation is listed:

| Aspect | Çiçek 2016 | Our seed | Why |
| ----- | ----- | ----- | ----- |
| Architecture, norm, activation, pooling | 3 poolings, BN before ReLU, max-pool, up-conv | Identical | Faithful |
| Loss | Weighted softmax cross-entropy (weights zero out unlabelled voxels) | Unweighted softmax cross-entropy | Our labels are dense; no unlabelled voxels to mask |
| Optimiser | SGD with momentum | Adam, 1e-3, constant | Faithful SGD would pre-satisfy half of T6. Adam is the honest modern naive default. |
| Augmentation | Rotation, scaling, gray-value, B-spline elastic deformation | Flips + 90° rotations only | The paper's elastic deformation would largely pre-satisfy T8 |

Soft Dice loss is *not* from this paper — it comes from V-Net (Milletari et al.,
2016). Our seed using cross-entropy is therefore the faithful choice, and T5 is
"adds a Dice term," not "adds a CE term."

---

## Target list

A target counts as recovered if its detection rule fires on **any elite in the
final MAP-Elites archive**. Elites are the programs that occupy cells of
`database.island_feature_maps` at the last checkpoint (the MAP-Elites archive,
not the auxiliary fitness archive). Record iteration and cumulative evaluation
GPU-hours at first appearance across *all* evaluated programs. No partial credit.

| # | Target | Seed value | nnU-Net value | Detection rule (AST on the evolve block, plus runtime probes where noted) |
| --- | ----- | ----- | ----- | ----- |
| T1 | Normalisation | `BatchNorm3d` | `InstanceNorm3d` | Any encoder norm layer is `InstanceNorm3d` or `GroupNorm` |
| T2 | Activation | `ReLU` | `LeakyReLU(0.01)` | Any activation is `LeakyReLU`, `ELU`, or `GELU` |
| T3 | Downsampling | `MaxPool3d` | Strided `Conv3d` | No `MaxPool3d`/`AvgPool3d` in the encoder **and** ≥1 `Conv3d` with `stride>1` |
| T4 | Deep supervision | Single head | Auxiliary heads, weights ∝ 2⁻ˡ | `forward` returns ≥2 tensors **and** the loss consumes ≥2 of them |
| T5 | Loss | Cross-entropy only | CE + soft Dice | A Dice-family term is composed with the CE term |
| T6 | Optimiser + schedule | `Adam`, constant LR | `SGD(nesterov, momentum≈0.99)` + poly | Optimiser is `SGD` with `nesterov=True` **and** LR decays monotonically below 10% of initial |
| T7 | Patch sampling | Uniform random location | ≥⅓ patches forced foreground | Sampler conditions the patch centre on a non-background voxel with probability ≥0.2 |
| T8 | Augmentation | Flip + 90° rotation | Elastic, scaling, gamma, blur, low-res simulation, brightness/contrast | ≥3 transform families beyond flip/rotate present |

Off-list discoveries (attention gates, residual blocks, unusual losses, …) are
reported in their own category, not discarded.

**Success criterion.** ≥4/8 targets recovered *and* recovery exceeding both
controls (zero-shot §10.4 and random-mutation §10.5 of the spec).

---

## Scoring, splits, and fitness

* **Fitness returned to OpenEvolve:** mean Dice over classes 1–7 on the 10-case
  fitness split, after a fresh-seed re-evaluation. Implementation:
  `src.metrics.mean_foreground_dice`. Do not reimplement Dice.
* **Inner tuning:** Optuna TPE, learning rate only, exactly 3 trials + 1
  re-evaluation. LR ~ log-uniform(1e-4, 1e-1). Child studies are seeded with the
  parent's `best_lr`.
* **Proxy training:** 40 cases, 96³ patches, batch size 2, 4000 optimiser steps.
* **Test set:** the shared sealed 20 from `results/splits.json`, opened once for
  ≤3 candidates.
* **HD95 empty-class penalty:** 100.0 mm, documented in `config.HD95_EMPTY_PENALTY_MM`.

Split files are generated once with seed 42, reconstruction-balanced (20+20 /
10+10 / 10+10 mial/irtk) and ICV-tertile balanced. Track B subdivides the 20-case
tuning split into 10 Optuna-selection + 10 fitness, 5 mial + 5 irtk each.
SHA-256 hashes of those JSON files are recorded in the pre-registration commit
message and in `results/split_hashes.json`.

---

## Controls

1. **Zero-shot LLM control (run first).** Same model, seed program, one prompt:
   *"Improve this 3D U-Net for segmentation of a small (n≈40) 3D medical imaging
   dataset."* No evolution, no fitness, no feedback. 20 independent samples at
   the evolution temperature. Score against the target list.
2. **Random-mutation control.** Same operators and prompts, fitness-blind
   selection: sample parents uniformly, keep every child. 60 iterations.
   Compare recovery *per iteration*.

---

## Archive feature axes

MAP-Elites grid: `params_millions` × `worst_class_dice`, 6 bins each (36 cells).
Evaluators return raw continuous values; OpenEvolve bins them.

---

## OpenEvolve compatibility notes (pre-registered)

* Upstream 0.3.2 has no `allow_full_rewrites` key. Diff-only evolution is
  `diff_based_evolution: true`.
* Claude Code CLI does not honor `temperature` or `max_tokens` from config; they
  are recorded for provenance only.
* Upstream `evaluate(program_path)` cannot see the parent's `best_lr`. A
  version-checked compatibility patch writes a sidecar JSON next to the temp
  program file. The patch hash is stamped in every `run_manifest.json`.

---

## Smoke profile

A `smoke` settings profile exists for local MPS debugging. It is
**not scientific**. Production numbers, VRAM gates, calibration, and paper
claims use the CUDA A6000 profile only.

---

## Amendments

### 2026-08-15 — A1: shared metric implementation

Track B will use the team-owned `src/metrics.py::compute_metrics` implementation
for proxy fitness and final evaluation. This replaces the provisional Track B
metric implementation so OpenEvolve and manually implemented models are scored
by the same code.

Consequences:

* Primary fitness remains mean foreground Dice over classes 1–7; the search
  objective is unchanged. The earlier named
  `src.metrics.mean_foreground_dice` reference is superseded by
  `src.metrics.compute_metrics(...)[\"mean\"][\"dice\"]`.
* Per-class Dice is read from the shared result to compute worst-class Dice.
* Secondary HD95, volume-similarity, and Euler-difference reports inherit the
  shared implementation exactly.
* The one-empty-mask HD95 penalty is therefore 374 mm, replacing the
  provisional 100 mm value.
* `src/metrics.py` and the Track B adapter are frozen by SHA-256 before a run.

This amendment was recorded before any paid LLM call or production evolution.
