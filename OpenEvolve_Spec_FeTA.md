# **Track B — OpenEvolve Implementation Specification**

**Owners:** Viet, James
**Hardware:** 1 × RTX A6000 (48 GB), dedicated
**Companion:** `Project_Pipeline_FeTA.md` (Track A — shared splits and metrics)

> **For the implementing agent.** §5–§9 are build instructions. Follow the phase order in
> §13; each phase has an acceptance test that must pass before proceeding. Where a value
> is given (channel counts, patch size, parameter ranges), it is normative — do not
> substitute a "better" default.

---

## **1. Research question and claim structure**

> **Q.** Starting from the original 3D U-Net (Çiçek et al., MICCAI 2016), can LLM-guided
> evolutionary code search rediscover the design decisions that distinguish it from
> nnU-Net (Isensee et al., *Nature Methods* 2021)?

This is deliberately **not** "can OpenEvolve beat nnU-Net." nnU-Net is the output of a
large, careful, completed search over this exact space, and FeTA has plateaued near
inter-annotator agreement. A beat-the-baseline framing yields a null result with no
information content. The rediscovery framing yields a measurable quantity — *target
recovery as a function of compute* — regardless of final Dice.

| Claim | Evidence required |
| ----- | ----- |
| Evolution recovers design decision *X* | *X* fires on an archive elite per the §3 detection rule |
| Recovery required **search**, not recall | Zero-shot LLM control (§10.1) did not produce *X* |
| Recovery required **fitness**, not drift | Random-mutation control (§10.2) did not produce *X* |
| Recovery was **worth the compute** | GPU-hours to first appearance, per target |

⚠️ The likeliest reviewer objection is contamination — the LLM has read both papers.
§10.1 is the control that makes the claim meaningful. Run it **first**.

---

## **2. Fixed decisions**

| Decision | Value |
| ----- | ----- |
| Seed | Çiçek 2016 architecture + deliberately minimal training recipe (§5.1) |
| Iterations | 150 |
| Islands | 3, migration every 25 iterations |
| Evolution mode | Diff-based; full rewrites disabled |
| Inner tuning | Optuna, learning rate only, 3 trials + 1 re-evaluation |
| Fitness | Mean Dice over classes 1–7, fitness split, fresh seed |
| Test set | The shared sealed 20 from `results/splits.json`, opened once for ≤3 candidates |

---

## **3. Pre-registration — commit before iteration 1**

Write `openevolve/PREREGISTRATION.md`, commit it, **do not edit**. Later changes go in a
dated amendment section at the bottom.

### **3.1 Target list**

| # | Target | Seed value | nnU-Net value | Detection rule (AST on the evolve block) |
| --- | ----- | ----- | ----- | ----- |
| T1 | Normalisation | `BatchNorm3d` | `InstanceNorm3d` | Any encoder norm layer is `InstanceNorm3d` or `GroupNorm` |
| T2 | Activation | `ReLU` | `LeakyReLU(0.01)` | Any activation is `LeakyReLU`, `ELU`, or `GELU` |
| T3 | Downsampling | `MaxPool3d` | Strided `Conv3d` | No `MaxPool3d`/`AvgPool3d` in the encoder **and** ≥1 `Conv3d` with `stride>1` |
| T4 | Deep supervision | Single head | Auxiliary heads, weights ∝ 2⁻ˡ | `forward` returns ≥2 tensors **and** the loss consumes ≥2 of them |
| T5 | Loss | Cross-entropy only | CE + soft Dice | A Dice-family term is composed with the CE term |
| T6 | Optimiser + schedule | `Adam`, constant LR | `SGD(nesterov, momentum≈0.99)` + poly | Optimiser is `SGD` with `nesterov=True` **and** LR decays monotonically below 10% of initial |
| T7 | Patch sampling | Uniform random location | ≥⅓ patches forced foreground | Sampler conditions the patch centre on a non-background voxel with probability ≥0.2 |
| T8 | Augmentation | Flip + 90° rotation | Elastic, scaling, gamma, blur, low-res simulation, brightness/contrast | ≥3 transform families beyond flip/rotate present |

### **3.2 Also pre-register**

* **Scoring.** A target counts as recovered if its rule fires on **any elite in the final
  MAP-Elites archive**. Record iteration and cumulative GPU-hours at first appearance.
  No partial credit.
* **Off-list discoveries.** Evolution will find things nnU-Net does not do (attention
  gates, residual blocks, unusual loss terms). Decide **now** that these are reported in
  their own category, not discarded — otherwise the analysis becomes post-hoc target
  fitting.
* **Success criterion.** ≥4/8 targets recovered *and* recovery exceeding both controls.

### **3.3 The system message must not leak the answer key**

`system_message` may describe the **task and its failure modes**. It may **not** name
instance normalisation, deep supervision, SGD, foreground oversampling, or any other
target by name or unambiguous paraphrase. Draft it, then have someone who has not read
§3.1 check for leakage. This is the easiest way to void the experiment.

---

## **4. Repository layout**

```
openevolve/
├── PREREGISTRATION.md          # §3, immutable after commit
├── settings.py                 # §4.1 — THE knob file. Hand-edited.
├── generate_config.py          # §4.1 — settings.py -> config.yaml
├── config.yaml                 # §11 — GENERATED. Do not hand-edit.
├── initial_program.py          # §5 — the ONLY file the LLM edits
├── evaluator.py                # §9
├── harness/                    # FROZEN. LLM never sees or edits these.
│   ├── __init__.py
│   ├── prepare.py              # §6 — one-off preprocessing to .npy cache
│   ├── data.py                 # §7.1 — case store + batch iterator
│   ├── train_loop.py           # §7.2 — generic loop over the contract
│   ├── infer.py                # §7.3 — sliding-window inference
│   └── guards.py               # §7.4 — anti-reward-hacking assertions
├── controls/
│   ├── zero_shot.py            # §10.1
│   └── random_mutation.py      # §10.2
├── analysis/
│   ├── score_targets.py        # §3.1 detection rules
│   ├── calibrate.py            # §8
│   └── figures.py
└── results/
    ├── trackB_subsplits.json
    ├── calibration.json
    ├── openevolve_output/
    └── target_scores.json
```

### **4.1 `settings.py` — the single knob file**

Every value a human might want to change lives here, in one hand-editable Python file
with inline comments. Nothing else in the repository hard-codes a tunable number.

**`config.yaml` is generated from it, not hand-edited.** `generate_config.py` reads
`settings.py` and writes `config.yaml`; `evaluator.py` imports `settings.py` directly.
If both were hand-edited they would drift, and no one could tell afterwards which
combination produced a given result.

**Contents:** search budget (iterations, islands, migration interval, population,
archive feature dimensions and bin counts); LLM configuration (provider, model weights,
temperature, per-model USD caps); the proxy definition (training case count, patch size,
batch size, optimiser steps, gradient clip); the inner Optuna block (number of trials,
pruner, and **the searchable parameter list itself** — declared as a list of
`(name, distribution, low, high)` tuples so a parameter can be added or removed without
touching `evaluator.py`); paths; and concurrency.

**Rules that keep it trustworthy:**

* **Validate on import.** Assert patch size divisible by 8, trial count ≥ 1, island count
  ≥ 1, weights summing to 1. A typo should fail immediately, not after six GPU-hours.
* **Stamp every run.** `evaluator.py` writes the full resolved settings dict plus the git
  commit hash into `results/openevolve_output/run_manifest.json` at startup. Every result
  is then traceable to the exact configuration that produced it.
* **Pre-registered values are not knobs.** Anything fixed in `PREREGISTRATION.md` — the
  splits, the fitness metric, the target list — does not belong here. Put those constants
  in `harness/`, where changing them requires a deliberate edit to frozen code.
* **Do not put the `system_message` here.** It lives in a separate
  `prompts/system_message.txt` reviewed for target leakage per §3.3. Burying it in a
  settings file makes it easy to edit casually, which is exactly what must not happen.

⚠️ Changing `settings.py` mid-run invalidates the run. Any edit after iteration 1 means
starting over or reporting the two segments separately.

---

## **5. `initial_program.py` — the seed**

### **5.1 Fidelity statement (goes verbatim into PREREGISTRATION.md)**

The **architecture** is faithful to Çiçek et al. 2016. The **training recipe** is
deliberately minimal, because the paper's own recipe would pre-satisfy two targets.
Every deviation is listed:

| Aspect | Çiçek 2016 | Our seed | Why |
| ----- | ----- | ----- | ----- |
| Architecture, norm, activation, pooling | 3 poolings, BN before ReLU, max-pool, up-conv | **Identical** | Faithful |
| Loss | Weighted softmax cross-entropy (weights zero out unlabelled voxels) | **Unweighted** softmax cross-entropy | Our labels are dense; no unlabelled voxels to mask |
| Optimiser | SGD with momentum | **Adam, 1e-3, constant** | Faithful SGD would pre-satisfy half of T6. Adam is the honest modern naive default. |
| Augmentation | Rotation, scaling, gray-value, B-spline elastic deformation | **Flips + 90° rotations only** | The paper's elastic deformation would largely pre-satisfy T8 |

> Note that soft Dice loss is *not* from this paper — it comes from V-Net (Milletari et
> al., 2016). Our seed using cross-entropy is therefore the faithful choice, and T5 is
> "adds a Dice term," not "adds a CE term."

### **5.2 Architecture — normative layer table**

Four resolution levels, three downsamplings. Input `[B,1,96,96,96]`, output
`[B,8,96,96,96]`.

| Stage | Operation | In → Out channels | Spatial |
| ----- | ----- | ----- | ----- |
| `enc0` | Conv3×3×3 → BN → ReLU → Conv3×3×3 → BN → ReLU | 1 → 32 → 64 | 96³ |
| pool | MaxPool3d(2) | 64 | 48³ |
| `enc1` | same block | 64 → 64 → 128 | 48³ |
| pool | MaxPool3d(2) | 128 | 24³ |
| `enc2` | same block | 128 → 128 → 256 | 24³ |
| pool | MaxPool3d(2) | 256 | 12³ |
| `bottom` | same block | 256 → 256 → 512 | 12³ |
| `up2` | ConvTranspose3d(k=2, s=2) | 512 → 512 | 24³ |
| `dec2` | concat with `enc2` (256), then block | 768 → 256 → 256 | 24³ |
| `up1` | ConvTranspose3d(k=2, s=2) | 256 → 256 | 48³ |
| `dec1` | concat with `enc1` (128), then block | 384 → 128 → 128 | 48³ |
| `up0` | ConvTranspose3d(k=2, s=2) | 128 → 128 | 96³ |
| `dec0` | concat with `enc0` (64), then block | 192 → 64 → 64 | 96³ |
| `out` | Conv1×1×1 | 64 → 8 | 96³ |

All 3×3×3 convolutions use `padding=1` and `bias=False` (BatchNorm supplies the shift).
Channels double *before* each pooling, per the paper, to avoid a bottleneck. Kaiming
normal initialisation with `nonlinearity='relu'`; BN weight 1, bias 0.

**Expected parameter count: ≈ 19.1 M.** Assert `18.5e6 < n_params < 19.5e6` in the seed's
acceptance test. If you are outside this range, the channel schedule is wrong.

96 / 2³ = 12, so the patch divides evenly through all three poolings. If you change patch
size, it must remain divisible by 8.

### **5.3 Module contract — frozen, outside the evolve block**

The harness calls exactly these six functions. Signatures may not change; the system
message states this explicitly and stage 1 enforces it.

```python
def build_model(in_channels: int, num_classes: int) -> torch.nn.Module
    # forward(x: FloatTensor[B,1,D,H,W]) -> FloatTensor[B,8,D,H,W]
    #                                    OR list/tuple of such tensors,
    #                                       index 0 = full resolution (deep supervision)

def build_loss() -> Callable[[Any, torch.LongTensor], torch.Tensor]
    # (model_output, target[B,D,H,W] int64 in 0..7) -> scalar tensor

def build_optimizer(params: Iterable, lr: float) -> torch.optim.Optimizer

def build_scheduler(optimizer, total_steps: int) -> Optional[_LRScheduler]
    # total_steps is the PROXY horizon (~4000), not 250k. Stepped once per
    # optimizer step. May return None for a constant learning rate.

def build_sampler() -> Callable[[np.ndarray, Tuple[int,int,int], np.random.Generator],
                                Tuple[int,int,int]]
    # (label_volume[D,H,W] int64, patch_size, rng) -> patch origin (d0,h0,w0)
    # Harness guarantees volume.shape >= patch_size on every axis.

def build_augmentation() -> Callable[[np.ndarray, np.ndarray, np.random.Generator],
                                     Tuple[np.ndarray, np.ndarray]]
    # (image_patch[D,H,W] float32, label_patch[D,H,W] int64, rng)
    #   -> (image_patch, label_patch), same shapes, contiguous
```

### **5.4 Reference implementation**

```python
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
```

### **5.5 Seed acceptance test** (`tests/test_seed.py`)

All must pass before the seed is used:

1. `build_model(1, 8)` instantiates; `18.5e6 < sum(p.numel()) < 19.5e6`.
2. Forward on `[2,1,96,96,96]` returns `[2,8,96,96,96]`, all finite.
3. Initial cross-entropy loss against random labels is within `ln(8) ± 0.15` ≈
   `2.079 ± 0.15`. **If this fails, the initialisation or class handling is broken** —
   debug it now, not after three days of training runs.
4. `loss.backward()` gives finite gradients on ≥99% of parameters.
5. Peak VRAM at `[2,1,96,96,96]` with autograd is **< 14 GB** (so three candidates fit
   concurrently on the A6000).
6. Overfit test: 200 steps on a **single** patch drives training Dice above 0.95. This is
   the definitive check that the loop, loss, and label handling agree.
7. `build_augmentation()` output shapes equal input shapes; label dtype is preserved and
   contains only values present in the input.

---

## **6. `harness/prepare.py` — one-off preprocessing (frozen)**

Preprocessing is **outside the search space**. Run once; cache to `.npy` on local SSD.

Per subject:

1. Load `T2w` and `dseg` with `nibabel`. Round `dseg` to `int64` — it is stored as
   `float32` and equality checks silently fail otherwise. Assert values ⊆ {0…7}.
2. **Resample to 0.5 mm isotropic.** Native spacing varies 0.43–0.70 mm across subjects.
   Trilinear for the image, **nearest-neighbour** for labels.
3. **Crop to the brain bounding box** derived from **non-zero image intensity** (not from
   labels — labels are unavailable at inference), plus a 16-voxel margin. FeTA
   reconstructions contain only the fetal brain, so background is exactly zero and this
   is reliable. Typically reduces 256³ to roughly 160³ and is the single biggest speed
   win in the pipeline.
4. **Pad** so every axis is ≥ 96 and divisible by 8.
5. **Z-score normalise** using mean and standard deviation of **non-zero voxels only**.
6. Save `{sid}_img.npy` (float32), `{sid}_lbl.npy` (int64), and a `meta.json` with
   original spacing, affine, crop offsets, and pre-pad shape — needed to map predictions
   back for Track A comparison.

Acceptance: round-trip one subject through crop/pad and back; assert the restored label
volume is bitwise identical to the input.

---

## **7. `harness/` — frozen modules**

### **7.1 `data.py`**

```python
class CaseStore:
    def __init__(self, case_ids: List[str], cache_dir: str)
        # Loads all volumes into RAM. 40 cases at ~160^3 float32 is ~2.6 GB. Fine.
    def volumes(self) -> List[Tuple[np.ndarray, np.ndarray]]

def batch_iterator(store, sampler, augment, patch_size=(96,96,96),
                   batch_size=2, steps=4000, seed=0) -> Iterator[Tuple[Tensor, Tensor]]
```

Per step: pick `batch_size` volumes uniformly at random; call `sampler` for each origin;
crop; call `augment`; stack; move to GPU as `float32` image `[B,1,D,H,W]` and `int64`
label `[B,D,H,W]`.

**Guards inside the iterator** (candidates control `sampler` and `augment`, so validate
their output rather than trusting it):
* Origin is in range on every axis, else clip and count the violation.
* Post-augmentation shapes equal `patch_size`, else raise.
* Label values remain in {0…7}, else raise.
* Wall-clock per step is tracked; a candidate whose augmentation exceeds 2 s/step is
  failed for timeout rather than being allowed to consume the whole budget.

`np.random.Generator` seeded per run — the same seed must reproduce the same batches.

### **7.2 `train_loop.py`**

```python
def train(recipe, case_ids, lr, total_steps, seed, cache_dir) -> nn.Module
```

1. Seed `torch`, `numpy`, `random`; set `torch.backends.cudnn.deterministic = True`.
2. `model = recipe.build_model(1, 8).cuda()`; `loss_fn = recipe.build_loss()`;
   `opt = recipe.build_optimizer(model.parameters(), lr)`;
   `sched = recipe.build_scheduler(opt, total_steps)`.
3. Loop `total_steps`: forward, loss, `backward`, **gradient clipping at norm 12**
   (nnU-Net's value; prevents one unstable candidate from producing NaNs that look like
   an architecture failure), `opt.step()`, `sched.step()` if not None.
4. Mixed precision via `torch.amp.autocast` + `GradScaler`. If a candidate produces
   NaN loss for 50 consecutive steps, abort and return `None` — the evaluator scores
   this 0 and reports the reason through artifacts.
5. Log loss every 100 steps into a compact summary returned alongside the model.

### **7.3 `infer.py`**

```python
def predict(model, image_volume, patch_size=(96,96,96), overlap=0.5) -> np.ndarray
```

Sliding window with **Gaussian importance weighting** (σ = patch/8) so patch seams do not
create artefacts. Accumulate softmax probabilities into a float32 buffer, divide by the
weight map, `argmax` → `int64` label volume. If the model returns a tuple, use index 0.
No test-time augmentation and no mirroring — that would be a free win unrelated to the
evolved design.

### **7.4 `guards.py`**

```python
def verify_frozen_files() -> None        # SHA-256 of harness/*, evaluator.py, src/metrics.py
def check_imports(program_path) -> None  # AST import whitelist
def assert_split_integrity(...) -> None  # case IDs match splits.json; sets disjoint
def assert_prediction_sane(pred) -> None # non-degenerate output
```

**Import whitelist** (checked by AST *before* any execution): `torch`, `numpy`, `math`,
`random`, `typing`, `scipy.ndimage`. **Forbidden:** `nnunetv2`, `monai`,
`segmentation_models_pytorch`, `timm`, `torch.hub`, `urllib`, `requests`, `subprocess`,
`os.system`, and any pretrained-weight loading.

> ⚠️ `import nnunetv2` is a one-line trivial win that scores beautifully and voids the
> entire experiment. Enforce this by AST check, not by hoping.

**Prediction sanity:** ≥5 distinct labels present; no class occupies >90% or 0% of
predicted foreground; output shape matches input; dtype is `int64`.

---

## **8. Data splits**

Import `results/splits.json` from Track A and subdivide its 20-case tuning split:

| Subset | n | Use |
| ----- | --- | ----- |
| Proxy training | 40 | Train every candidate |
| Optuna selection | 10 | Inner learning-rate selection only |
| Fitness | 10 | The number returned to OpenEvolve |
| **Sealed test** | 20 | Final ≤3 candidates only |

The 10/10 subdivision is itself balanced on reconstruction method (5 mial + 5 irtk each).
Write once to `results/trackB_subsplits.json`.

Separating selection from fitness is not pedantry: reusing one set for both makes "best
of 3 trials" a maximum computed on the scoring data, inflating every candidate by roughly
one noise standard deviation — larger than the effects being measured.

---

## **9. `evaluator.py`**

### **9.1 Stage 1 — smoke test (~30 s, no training)**

Expect this to reject 30–40% of proposals.

1. `guards.verify_frozen_files()`, then `guards.check_imports()`.
2. All six contract functions present with correct arity.
3. Model instantiates; parameter count in [1e6, 1.5e8].
4. Forward on `[2,1,96,96,96]` gives correct shape (or a sequence whose element 0 does);
   output finite.
5. Loss finite and scalar; `backward()` gives finite gradients on ≥90% of parameters.
6. Peak VRAM < 14 GB.
7. No bare `except:` inside the evolve block (a candidate that swallows exceptions can
   look healthy while doing nothing).

On failure return `{"stage1_ok": 0.0}` with the reason in `artifacts["stderr"]`. The
artifact channel feeds the error into the next generation's prompt, which is what stops
the LLM re-proposing the same broken layer.

### **9.2 Stage 2 — fitness (~60–75 min)**

```
for trial in 3 Optuna trials (TPE, enqueued with the parent's best_lr):
    lr ~ log-uniform(1e-4, 1e-1)
    model = train(recipe, PROXY_40, lr, PROXY_STEPS, seed=1000+trial)
    score = mean Dice, classes 1-7, on the 10 OPTUNA-SELECTION cases
best_lr = argmax(score)

# Re-evaluation — this is the reported fitness
model = train(recipe, PROXY_40, best_lr, PROXY_STEPS, seed=7777)
fitness = mean Dice, classes 1-7, on the 10 FITNESS cases
```

**Why re-evaluate.** `max` over T trials is biased: for noise σ, `E[max of T] ≈ μ +
σ·c(T)`, with `c(3) ≈ 0.85`. At σ ≈ 0.7 Dice that is ~0.6 Dice of inflation. The fresh
seed on held-out cases removes it, at a cost of one extra run in four.

**Fixed trial count.** Never 3 for one candidate and 8 for another — unequal budgets make
archive fitness incomparable.

**LR inheritance.** Persist `best_lr` on the program record; seed the child's study with
`study.enqueue_trial({"lr": parent_best_lr})`. Children are small diffs of parents, so
optima are nearby — this is what makes 3 trials sufficient.

Dice comes from `src/metrics.py` (Sonia's module). Do not reimplement it; Track A and
Track B numbers must be produced by identical code.

### **9.3 Returned metrics**

```python
return EvaluationResult(
    metrics={
        "combined_score": fitness,
        "worst_class_dice": min(per_class),
        "params_millions": n_params / 1e6,
        "best_lr": best_lr,
        **{f"dice_class_{i}": d for i, d in enumerate(per_class, start=1)},
    },
    artifacts={"stderr": captured_stderr, "train_curve": loss_summary},
)
```

Return **raw** feature values — OpenEvolve does the binning.

### **9.4 Reward-hacking guards**

| Threat | Guard |
| ----- | ----- |
| Editing evaluator or metrics | SHA-256 verification before every run |
| Shrinking or reordering the fitness set | Case IDs asserted against `splits.json` |
| Training on fitness or test data | Read-only mounts; disjointness assertion |
| Constant or degenerate output | `assert_prediction_sane` |
| Caching across candidates | Fresh process per evaluation; scratch wiped |
| Swallowing exceptions | Bare `except` fails stage 1; harness owns error handling |

Run inside the OpenEvolve Docker image, not the host environment. **Read the top-5
candidates' diffs by hand.** Always.

---

## **10. Calibration and controls**

### **10.1 Proxy configuration**

| Parameter | Value |
| ----- | ----- |
| Training cases | 40 |
| Patch size | 96³ |
| Batch size | 2 |
| Optimiser steps | 4,000 |
| Target wall clock | 12–15 min |

### **10.2 Noise floor**

Run the unmodified seed 5× with different seeds; compute σ of fitness. Expect 0.5–1.0
Dice on 10 cases. **Write the number where the team can see it.** Improvements below ~2σ
are nothing.

### **10.3 Rank-correlation gate — the go/no-go**

Hand-build 8 variants spanning good to bad: seed; +InstanceNorm; +Dice term;
+deep supervision; LR 10× too high; base 8 channels; +SGD/poly; +foreground sampling.
Score each under the proxy **and** under full training (1000-epoch equivalent, 5-fold).
Spearman ρ between rankings:

| ρ | Action |
| --- | ----- |
| ≥ 0.7 | Proceed as specified |
| 0.6 – 0.7 | Proceed; report proxy fidelity as a limitation |
| < 0.6 | **Stop.** Escalate to 8,000 steps at 128³, cut to 80 iterations, re-gate |

~40 GPU-hours, and the highest-value expenditure in the project. Without it, everything
downstream is an expensive random number generator.

### **10.4 Zero-shot LLM control — run before the main run**

Give the same model the seed and one prompt: *"Improve this 3D U-Net for segmentation of
a small (n≈40) 3D medical imaging dataset."* No evolution, no fitness, no feedback. 20
independent samples at the evolution temperature. Score against §3.1.

Cost: ~$5, one hour. The outcome determines the paper's framing:
* 6–8 of 8 named → claim narrows to *"search validates priors the model already holds;
  targets T*x*, T*y* required empirical selection."* Still publishable, and more honest
  than most NAS papers.
* 2–3 of 8 named → the discovery claim is strong.

### **10.5 Random-mutation control**

Same operators, same prompts, but fitness-blind selection: sample parents uniformly, keep
every child. Isolates whether the fitness signal does work. Run at 60 iterations and
compare recovery *per iteration*.

---

## **11. `config.yaml` (generated)**

⚠️ **This file is written by `generate_config.py` from `settings.py` (§4.1). Do not edit
it by hand** — your change will be overwritten on the next generation and the run
manifest will disagree with what actually ran. The block below is the expected *output*,
shown so the mapping from `settings.py` is legible.

⚠️ OpenEvolve moves fast. **Verify every key against `configs/default_config.yaml` in the
installed version** before the first run, and update `generate_config.py` if names have
shifted.

```yaml
max_iterations: 150
random_seed: 42
checkpoint_interval: 10

llm:
  provider: "claude_code"
  models:
    - name: "sonnet"
      weight: 0.8
      max_tokens: 16000
      max_budget_usd: 40.0
    - name: "opus"
      weight: 0.2
      max_tokens: 16000
      max_budget_usd: 30.0
  temperature: 0.8

prompt:
  num_top_programs: 3
  num_diverse_programs: 2
  include_artifacts: true
  system_message: |
    You are an expert in deep learning for 3D medical image segmentation.

    TASK: improve a 3D U-Net that segments fetal brain MRI into 7 tissue classes
    plus background.

    DATASET CHARACTERISTICS:
    - 40 training volumes. This is a very small dataset.
    - Input patches are 96x96x96 voxels, batch size 2. Neither can change.
    - Volumes are mostly background; brain occupies a minority of the volume.
    - Two tissue classes are very small and are the hardest to segment.
    - Tissue contrast varies systematically across the cohort because the subjects
      are at different developmental stages.
    - Reference annotations are imperfect: drawn on every second or third slice in
      one plane, then interpolated.

    MUST NOT CHANGE:
    - The six function signatures in the module contract.
    - Patch size, batch size, or the number of output classes.
    - Anything outside the EVOLVE-BLOCK markers.

    ALLOWED:
    - Network architecture, normalisation, activations, connectivity.
    - Loss function composition.
    - Optimiser and learning-rate schedule (use the total_steps argument).
    - Patch sampling strategy.
    - Data augmentation.

    Make ONE focused change per proposal and briefly explain your reasoning.

database:
  population_size: 120
  num_islands: 3
  migration_interval: 25
  feature_dimensions: ["params_millions", "worst_class_dice"]
  feature_bins:
    params_millions: 6
    worst_class_dice: 6

evaluator:
  timeout: 7200
  parallel_evaluations: 3
  cascade_evaluation: true
  cascade_thresholds: [0.35]
  enable_artifacts: true
  use_llm_feedback: false

diff_based_evolution: true
allow_full_rewrites: false
max_code_length: 20000
```

* **6×6 = 36 archive cells** at 150 iterations. Larger grids leave most cells empty and
  quality-diversity stops functioning.
* **`worst_class_dice` as an axis** preserves specialists — candidates with mediocre mean
  Dice but good brainstem or deep-GM performance. Standard selection would discard
  exactly the interesting mutations.
* **`use_llm_feedback: false`** — LLM code-quality scoring would contaminate fitness with
  a prior about what good architecture looks like, which is the thing under test.

---

## **12. Budget**

| Component | GPU-hours |
| ----- | ----- |
| Calibration (noise floor + 8 variants, both regimes) | 45 |
| Zero-shot control | 0 (LLM only, ~$5) |
| Main evolution: 150 × 4 runs × ~14 min | 140 |
| Random-mutation control: 60 iterations | 56 |
| Final full training, top 3 + seed + controls | 60 |
| **Total** | **~300** |

**Concurrency reality check.** Three concurrent jobs will *not* give 3× throughput — a
batch-2 3D convolution job leaves the A6000 underutilised, so expect 1.8–2.2×. At 2× that
is ~150 hours wall clock ≈ **6–7 days continuous**, plus calibration. Plan two weeks
including failures.

If overrunning, cut the random-mutation control to 40 iterations before cutting main-run
iterations — recovery-per-iteration is the comparison, so a shorter control still works.

---

## **13. Build order (for the implementing agent)**

Each phase has a gate. Do not proceed past a failing gate.

| Phase | Build | Gate |
| --- | ----- | ----- |
| **P0** | `PREREGISTRATION.md` committed; `settings.py` + `generate_config.py` | Leakage check by someone who has not read §3.1; settings validation raises on a deliberately bad value |
| **P1** | `harness/prepare.py`, run on all 80 subjects | Crop/pad round-trip is bitwise identical; cache size sane |
| **P2** | `initial_program.py` (§5.4) | All 7 acceptance tests in §5.5 pass, especially the `ln(8)` and single-patch overfit checks |
| **P3** | `harness/data.py`, `train_loop.py`, `infer.py` | Same seed reproduces identical batches; 100-step run completes; sliding-window output shape matches input |
| **P4** | `harness/guards.py` | Write 5 deliberately malicious candidates (imports nnU-Net; returns a constant; edits the split; bare `except`; fabricates output) — all 5 must be caught |
| **P5** | `evaluator.py` stage 1 only | Rejects all P4 malicious candidates in <60 s each |
| **P6** | `evaluator.py` stage 2 | Seed evaluates end-to-end in 60–75 min; fitness is plausible (>0.5 Dice) |
| **P7** | `controls/zero_shot.py` | 20 samples scored; result recorded before the main run |
| **P8** | `analysis/calibrate.py` | Noise floor σ measured; Spearman ρ ≥ 0.6 (§10.3) |
| **P9** | `config.yaml`, 5-iteration dry run | Checkpointing works; archive populates; artifacts reach the prompt |
| **P10** | Full 150-iteration run | — |
| **P11** | `analysis/score_targets.py`, random-mutation control, final training | — |
| **P12** | Sealed test opened once; figures; write-up | — |

**Hard dependencies:** `src/metrics.py` (Sonia) and `results/splits.json` (Caolan) must
exist before P6. Both are Track A week-1 deliverables — confirm the `metrics.py` API with
Sonia directly rather than waiting for it to appear.

---

## **14. Deliverables**

**Figure 1 (headline).** Cumulative targets recovered (0–8) vs GPU-hours. Three traces:
evolution, random-mutation control, zero-shot control (a horizontal line — it consumes no
GPU time).

**Figure 2.** Best-so-far fitness vs GPU-hours, with reference lines for the seed at full
training and stock nnU-Net, and a shaded ±2σ noise band.

**Figure 3.** MAP-Elites archive, `params_millions` × `worst_class_dice`, cells coloured
by mean Dice. Shows what quality-diversity retained that pure selection would have lost.

**Table 1.** Per target: recovered (Y/N), first iteration, GPU-hours to first appearance,
zero-shot hit rate, random-mutation hit rate.

**Table 2.** Sealed-test performance: seed, stock nnU-Net, `pengyy` config, top-3 evolved.
Mean Dice, per-class Dice, HD95, with significance tests from `src/validate.py`.

**Table 3.** Off-list discoveries: what evolution found that nnU-Net does not do, and
whether it survived into the final archive.

---

## **15. Known risks**

| Risk | Likelihood | Mitigation |
| ----- | ----- | ----- |
| Proxy ρ < 0.6 | Medium | P8 gate catches it in week 1; escalate proxy, cut iterations |
| Zero-shot control names most targets | **High** | Reframe as validation-vs-discovery (§10.4). Plan for it. |
| Evolution stalls in a local optimum | Medium | 3 islands + QD archive + `num_diverse_programs: 2` |
| LLM repeats a broken change | Medium | Artifacts feed stderr back; add the pattern to the system message **without naming targets** |
| Reward hacking | Low–medium | §9.4 guards + P4 adversarial tests + manual diff review |
| Improvements within noise | **High** | Noise floor in every table; significance tests via `src/validate.py` |
| Contract drift breaks the harness | Low | `allow_full_rewrites: false`; stage-1 arity check |
