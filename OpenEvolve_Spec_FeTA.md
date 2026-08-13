# **Track B — OpenEvolve Specification**

**Owners:** Viet, James
**Hardware:** 1 × RTX A6000 (48 GB), dedicated
**Companion document:** `Project_Pipeline_FeTA.md` (Track A, shared splits and metrics)

---

## **1. Research question and claim structure**

> **Q.** Starting from the original 3D U-Net (Çiçek et al., 2016), can LLM-guided
> evolutionary code search rediscover the design decisions that distinguish it from
> nnU-Net (Isensee et al., 2021)?

This is deliberately **not** "can OpenEvolve beat nnU-Net." nnU-Net is the output of a
large, careful, completed search over this exact space, and the FeTA benchmark has
plateaued near inter-annotator agreement. A beat-the-baseline framing produces a null
result with no information content. The rediscovery framing produces a measurable
quantity — *target recovery rate as a function of compute* — whether or not the final
Dice exceeds the baseline.

### **Claim decomposition**

| Claim | Evidence required |
| ----- | ----- |
| Evolution recovers design decision *X* | *X* present in an archive elite, per §3 detection rule |
| Recovery required **search**, not recall | Zero-shot LLM control (§9.1) did not produce *X* |
| Recovery required **fitness**, not drift | Random-mutation control (§9.2) did not produce *X* |
| Recovery was **worth the compute** | GPU-hours to first appearance, per target |

A target recovered by the zero-shot control is not a discovery — it is retrieval from the
model's training data. Reporting the split between these two categories *is* the result.

⚠️ **The most likely reviewer objection is contamination:** the LLM has read both papers.
§9.1 is not optional; it is the control that makes the claim meaningful. Budget it first.

---

## **2. Design decisions fixed before any run**

| Decision | Value | Rationale |
| ----- | ----- | ----- |
| Seed program | Çiçek 2016 3D U-Net, faithful | Citable starting point, not a strawman |
| Iterations | 150 | ~4 days wall clock at achievable concurrency |
| Islands | 3, migration every 25 iterations | Prevents single-lineage collapse at this budget |
| Evolution mode | Diff-based, full rewrites disabled | Full rewrites break the module contract |
| Inner tuning | Optuna, LR only, 3 trials + 1 re-eval | See §7.3 |
| Fitness | Mean Dice over classes 1–7, fitness split, fresh seed | Same `metrics.py` as Track A |
| Test set | The shared sealed 20 from `results/splits.json` | Opened once, for ≤3 final candidates |

---

## **3. Pre-registration — commit this before iteration 1**

Write `openevolve/PREREGISTRATION.md`, commit it, and **do not edit it afterwards**. Any
later change goes in a dated amendment section at the bottom.

### **3.1 Target list**

Eight design decisions separating Çiçek 2016 from nnU-Net. Each has a mechanical
detection rule evaluated by AST inspection of the evolved program (`score_targets.py`).

| # | Target | Seed value | nnU-Net value | Detection rule |
| --- | ----- | ----- | ----- | ----- |
| T1 | Normalisation | `BatchNorm3d` | `InstanceNorm3d` | Any norm layer in the encoder is `InstanceNorm3d` or `GroupNorm` |
| T2 | Activation | `ReLU` | `LeakyReLU(0.01)` | Any activation is `LeakyReLU`, `ELU`, or `GELU` |
| T3 | Downsampling | `MaxPool3d` | Strided `Conv3d` | No `MaxPool3d`/`AvgPool3d` in the encoder **and** ≥1 `Conv3d` with `stride>1` |
| T4 | Deep supervision | Single output head | Auxiliary heads, weights ∝ 2⁻ˡ | Forward returns a tuple/list of ≥2 tensors **and** loss consumes ≥2 elements |
| T5 | Loss | Soft Dice only | Soft Dice + cross-entropy | Loss composes a Dice-family term with `cross_entropy`/`nll_loss` |
| T6 | Optimiser + schedule | `Adam`, constant LR | `SGD(nesterov, momentum=0.99)` + poly | Optimiser is `SGD` with `nesterov=True` **and** LR decays monotonically to <10% of initial |
| T7 | Patch sampling | Uniform random location | ≥⅓ patches forced foreground | Sampler conditions patch centre on a non-background voxel with probability ≥0.2 |
| T8 | Augmentation | Flip + 90° rotation | Elastic, scaling, gamma, blur, low-res simulation, brightness/contrast | ≥3 transform families beyond flip/rotate present in the pipeline |

### **3.2 Also pre-register**

* **Scoring rule.** A target counts as recovered if the rule fires on **any elite in the
  final MAP-Elites archive**. Record first iteration and cumulative GPU-hours at first
  appearance. Partial credit is not awarded.
* **Off-list discoveries.** Evolution will find things nnU-Net does not do (attention
  gates, residual blocks, alternative normalisations, unusual loss terms). These get
  their own reporting category. Decide **now** that they are reported, not discarded —
  otherwise the analysis becomes post-hoc target-fitting.
* **Success criterion.** State in advance what counts as a positive result. Suggested:
  ≥4/8 targets recovered *and* recovery rate exceeding both controls.

### **3.3 The system message must not leak the answer key**

The OpenEvolve `system_message` may describe the **task and its failure modes** — small
dataset, batch size 2, class imbalance with brainstem and cerebellum at ~1–2% of volume,
GM/WM contrast that weakens with maturation. It may **not** mention instance
normalisation, deep supervision, SGD, foreground oversampling, or any other target by
name or by unambiguous paraphrase.

Draft the message, then have someone who has not read §3.1 check it for leakage. This is
the single easiest way to void the entire experiment.

---

## **4. Repository layout**

```
openevolve/
├── PREREGISTRATION.md          # §3, committed before iteration 1, immutable
├── config.yaml                 # OpenEvolve configuration
├── initial_program.py          # Seed: vanilla 3D U-Net + training recipe
├── evaluator.py                # Fitness function (cascade + inner Optuna)
├── harness/
│   ├── __init__.py
│   ├── data.py                 # FROZEN. Loaders, patch extraction plumbing.
│   ├── train_loop.py           # FROZEN. Generic loop; consumes the recipe contract.
│   ├── infer.py                # FROZEN. Sliding-window inference.
│   └── guards.py               # FROZEN. Anti-reward-hacking assertions.
├── controls/
│   ├── zero_shot.py            # §9.1
│   └── random_mutation.py      # §9.2
├── analysis/
│   ├── score_targets.py        # AST detection rules from §3.1
│   ├── calibrate.py            # §8 noise floor + Spearman gate
│   └── figures.py
└── results/
    ├── calibration.json
    ├── openevolve_output/      # checkpoints, archive, logs
    └── target_scores.json
```

`harness/` is frozen: the LLM never sees it and never edits it. `guards.py` is
hash-verified by `evaluator.py` before every candidate evaluation.

---

## **5. Data splits**

Import `results/splits.json` from Track A. Track B subdivides the tuning split:

| Subset | n | Source | Use |
| ----- | --- | ----- | ----- |
| Proxy training | 40 | Track A training split | Train every candidate |
| Optuna selection | 10 | First half of Track A tuning split | Inner LR selection only |
| Fitness | 10 | Second half of Track A tuning split | The number returned to OpenEvolve |
| **Sealed test** | 20 | Track A test split | Final ≤3 candidates only |

Splitting Optuna selection from fitness is not pedantry. If the same 20 cases were used
for both, "best of 3 trials" would be a maximum computed on the scoring data, inflating
every candidate's fitness by roughly one noise standard deviation — larger than the
effects we are trying to detect.

The 10/10 subdivision must itself be balanced on reconstruction method (5 mial + 5 irtk
each). Write it to `results/trackB_subsplits.json` once.

---

## **6. `initial_program.py` — the seed**

### **6.1 Module contract (frozen, outside the evolve block)**

The harness calls exactly these five functions. Signatures may not change; the LLM is
told this explicitly in the system message.

```python
def build_model(in_channels: int, num_classes: int) -> torch.nn.Module: ...
    # forward(x: [B,1,D,H,W]) -> Tensor [B,C,D,H,W]
    #                          OR list/tuple of such tensors (deep supervision,
    #                          index 0 = full resolution)

def build_loss() -> Callable[[Any, torch.Tensor], torch.Tensor]: ...
    # (model_output, target[B,D,H,W] int64) -> scalar

def build_optimizer(params, lr: float) -> torch.optim.Optimizer: ...

def build_scheduler(optimizer, total_steps: int): ...
    # total_steps is the PROXY horizon, not 250k. See §6.3.

def build_sampler() -> Callable: ...
    # (label_volume, patch_size, rng) -> patch origin coordinates

def build_augmentation() -> Callable: ...
    # (image_patch, label_patch, rng) -> (image_patch, label_patch)
```

### **6.2 Seed implementation**

Faithful Çiçek 2016 with a plain training recipe:

* 3D U-Net, 5 stages, base 32 channels doubling per stage (32/64/128/256/320)
* Two 3×3×3 convolutions per stage, each followed by `BatchNorm3d` + `ReLU`
* `MaxPool3d(2)` downsampling, `ConvTranspose3d` upsampling, concatenating skips
* Single 1×1×1 output convolution to 8 channels
* Soft Dice loss, macro over classes 1–7
* `Adam(lr=1e-3)`, constant LR
* Uniform random patch location
* Random flips along all axes + 90° rotations only

Everything inside `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`. Imports and the
contract docstring sit outside.

### **6.3 The horizon-relative schedule trap**

`build_scheduler` receives `total_steps` = the proxy horizon (~4,000), not 250,000.

This matters directly for **T6**. SGD with Nesterov and poly decay is a slow-start,
strong-finish recipe; Adam converges faster early. A short proxy with an absolute-length
schedule systematically favours Adam, meaning **the experiment would structurally forbid
discovering T6**. Passing the true horizon lets a candidate write
`(1 - step/total_steps)**0.9` and get the correct schedule shape at any budget.

Residual bias remains — short horizons still flatter adaptive optimisers. State this in
the paper. The bias runs *against* the hypothesis, so discovering T6 anyway is stronger
evidence, not weaker.

---

## **7. `evaluator.py` — the fitness function**

### **7.1 Cascade stage 1 — smoke test (~30 s, no training)**

Rejects malformed candidates before they consume GPU time. Expect it to kill 30–40% of
proposals.

1. **Import whitelist** (AST-level, before execution). Permitted: `torch`, `numpy`,
   `math`, `random`, `typing`, `scipy.ndimage`. **Forbidden:** `nnunetv2`,
   `monai.networks.nets`, `segmentation_models_pytorch`, `torch.hub`, `timm`, any
   network downloading or pretrained-weight loading. Importing a pre-built nnU-Net is a
   trivial win and destroys the experiment; ban it explicitly rather than hoping.
2. All six contract functions exist with correct arity.
3. `build_model` instantiates; parameter count in [1e6, 1.5e8].
4. Forward pass on `[2,1,96,96,96]` returns the right shape (or a list whose element 0
   does); output is finite.
5. Loss is finite and scalar; `backward()` produces finite gradients on ≥90% of
   parameters.
6. Peak VRAM under 14 GB (so three candidates fit concurrently).

Return `{"stage1_ok": 0.0}` with the failure reason in `artifacts["stderr"]` on failure.
The artifact channel feeds the error back into the next generation's prompt, which is
what stops the LLM re-proposing the same broken layer.

### **7.2 Cascade stage 2 — the real fitness (~60–75 min)**

```
for trial in 3 Optuna trials (TPE, seeded with parent's best LR):
    lr ~ log-uniform(1e-4, 1e-1)
    train on 40 proxy cases for PROXY_STEPS, seed = 1000 + trial
    score mean Dice on the 10 Optuna-selection cases
best_lr = argmax

# Re-evaluation — this is the reported fitness
train on 40 proxy cases at best_lr, seed = 7777
fitness = mean Dice classes 1-7 on the 10 FITNESS cases
```

**Why the re-evaluation.** `max` over T trials is a biased estimator of candidate
quality: for noise σ, `E[max of T] ≈ μ + σ·c(T)`, with `c(3) ≈ 0.85`. At σ ≈ 0.7 Dice
that is ~0.6 Dice of pure inflation, comparable to real effects. The re-evaluation at a
fresh seed on held-out cases removes it. Cost: one extra run in four.

**Fixed trial count.** Never 3 trials for one candidate and 8 for another. Unequal
budgets make fitness incomparable across the archive.

**LR inheritance.** Persist `best_lr` on the program record and seed the child's study
with `study.enqueue_trial({"lr": parent_best_lr})`. Children are small diffs of parents,
so their optima are nearby; this is what makes 3 trials sufficient.

### **7.3 Returned metrics**

```python
return EvaluationResult(
    metrics={
        "combined_score": fitness,            # mean Dice, classes 1-7
        "worst_class_dice": min(per_class),   # MAP-Elites axis
        "params_millions": n_params / 1e6,    # MAP-Elites axis
        "best_lr": best_lr,
        **{f"dice_class_{i}": d for i, d in enumerate(per_class, start=1)},
    },
    artifacts={"stderr": captured_stderr, "train_curve": loss_summary},
)
```

Return **raw** values for feature dimensions — OpenEvolve handles binning.

### **7.4 Guards against reward hacking**

The LLM has code execution and an incentive to maximise a number. Assume adversarial
behaviour even without adversarial intent.

| Threat | Guard |
| ----- | ----- |
| Editing the evaluator or metrics | SHA-256 verify `evaluator.py`, `harness/*.py`, `src/metrics.py` before every run |
| Shrinking or reordering the fitness set | Case IDs asserted against `splits.json` after loading |
| Training on fitness or test data | Data mounted read-only; loader asserts disjointness of ID sets |
| Returning a constant or degenerate map | Assert no class occupies >90% or 0% of predicted foreground; assert ≥5 distinct labels present |
| Caching results across candidates | Fresh process per evaluation; scratch directory wiped |
| Swallowing exceptions to look healthy | Harness catches, never the candidate; bare `except` in the evolve block fails stage 1 |
| Fabricating outputs without a forward pass | Assert prediction tensor is on the model's device and `requires_grad` history is consistent |

Run inside the OpenEvolve Docker image, not your host environment. **Manually read the
top-5 candidates' diffs.** Always. The artifacts channel already surfaces stderr, so this
is cheap.

---

## **8. Calibration — the go/no-go gate (week 1, before anything else)**

The proxy is a deliberately degraded training run standing in for the real thing. It is
useful **only if it ranks candidates the way full training would.** Measure this before
committing 200 GPU-hours to it.

### **8.1 Proxy configuration to test**

| Parameter | Value |
| ----- | ----- |
| Training cases | 40 |
| Patch size | 96³ *(reduced from 128³)* |
| Batch size | 2 |
| Optimiser steps | 4,000 |
| Target wall clock | 12–15 min |

### **8.2 Noise floor**

Run the unmodified seed 5× with different seeds. Compute σ of fitness. Expect 0.5–1.0
Dice on a 10-case fitness split. **Write this number where the team can see it.** Any
evolved "improvement" below ~2σ is nothing.

### **8.3 Rank-correlation gate**

Hand-build 8 variants spanning good to bad — e.g. seed; seed+InstanceNorm; seed+CE·Dice;
seed+deep supervision; seed with LR 10× too high; seed with base 8 channels;
seed+SGD/poly; seed+foreground sampling. Score each under the proxy **and** under full
training (1000-epoch equivalent, 5-fold). Compute Spearman ρ between the rankings.

| ρ | Action |
| --- | ----- |
| ≥ 0.7 | Proceed as specified |
| 0.6 – 0.7 | Proceed; state proxy fidelity as a limitation |
| < 0.6 | **Stop.** Escalate proxy to 8,000 steps at 128³, cut to 80 iterations, re-gate |

This costs ~8 full training runs (~40 GPU-hours) and is the highest-value expenditure in
the project. Without it, everything downstream is an expensive random number generator.

---

## **9. Controls**

### **9.1 Zero-shot LLM control (run first — it may reframe the whole project)**

Give the same model the seed program and one prompt: *"Improve this 3D U-Net for
segmentation of a small (n≈40) 3D medical imaging dataset."* No evolution, no fitness, no
feedback. 20 independent samples at the evolution temperature. Score each against §3.1.

Cost: ~$5 and one hour. Do it **before** the main run, because the outcome changes the
paper's framing:

* Model names 6–8 of 8 → the claim narrows to *"search validates priors the model
  already holds; targets T*x*, T*y* required empirical selection."* Still publishable,
  and more honest than most NAS papers.
* Model names 2–3 of 8 → the discovery claim is strong.

### **9.2 Random-mutation control**

Same mutation operators and same iteration count, but selection is fitness-blind: sample
parents uniformly from the archive, accept every child. Isolates whether the fitness
signal does work, or whether LLM proposals drift toward nnU-Net regardless.

Cost reduction: run at 60 iterations, not 150, and compare recovery *per iteration*.

### **9.3 Reference points**

Both from Track A, no extra compute: stock nnU-Net 3d_fullres (the design target) and the
vanilla seed at full training (the floor).

---

## **10. `config.yaml`**

⚠️ OpenEvolve moves fast. **Verify every key against `configs/default_config.yaml` in the
installed version** before the first run; some names below may have shifted.

```yaml
max_iterations: 150
random_seed: 42
checkpoint_interval: 10

llm:
  provider: "claude_code"          # OAuth via Claude Code CLI, no API key
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
    - Input patches are 96x96x96 voxels, batch size 2. Batch size cannot change.
    - Volumes are mostly background; brain occupies a minority of the volume.
    - Two tissue classes occupy roughly 1-2% of brain volume each and are the
      hardest to segment.
    - Tissue contrast varies systematically across the cohort because the subjects
      are at different developmental stages.
    - Reference annotations are imperfect: they were drawn on every second or third
      slice in one plane and interpolated.

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

    Make ONE focused change per proposal and explain your reasoning briefly.

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

**Notes on the choices.**

* **6×6 = 36 archive cells** at 150 iterations. Larger grids leave most cells empty and
  the quality-diversity mechanism stops functioning.
* **`worst_class_dice` as an axis** deliberately preserves specialists — candidates with
  mediocre mean Dice but unusually good brainstem or deep-GM performance. Those are the
  interesting mutations and standard selection would discard them.
* **`use_llm_feedback: false`** — LLM code-quality scoring would contaminate fitness with
  a prior about what "good" architecture looks like, which is precisely the thing under
  test.
* **`cascade_thresholds: [0.35]`** — candidates below 0.35 mean Dice after stage 2 are
  not promoted. Sanity floor only; the seed should score well above this.

---

## **11. Budget**

| Component | GPU-hours | Note |
| ----- | ----- | ----- |
| Calibration (noise floor + 8 variants, both regimes) | 45 | Week 1, gated |
| Zero-shot control | 0 | LLM only, ~$5 |
| Main evolution: 150 × 4 runs × ~14 min | 140 | Stage-1 rejections reduce this ~30% in practice |
| Random-mutation control: 60 iterations | 56 | |
| Final full training, top 3 + seed + controls | 60 | 1000-epoch equivalent |
| **Total** | **~300** | |

**Concurrency reality check.** Three concurrent jobs on one A6000 will *not* give 3×
throughput — a batch-2 3D convolution job leaves the card underutilised, so expect
roughly 1.8–2.2× effective speedup. At 2× that is ~150 hours wall clock ≈ **6–7 days of
continuous compute**, plus calibration. Plan for two weeks including failures.

**Cheapest lever if you overrun:** cut the random-mutation control to 40 iterations
before cutting main-run iterations. Recovery-per-iteration is the comparison, so a
shorter control still supports the claim.

---

## **12. Known risks**

| Risk | Likelihood | Mitigation |
| ----- | ----- | ----- |
| Proxy ρ < 0.6 | Medium | §8 gate catches it in week 1; escalate proxy, cut iterations |
| Zero-shot control names most targets | **High** | Reframe claim as validation-vs-discovery (§9.1). Not a failure — plan for it. |
| Evolution stalls in a local optimum | Medium | 3 islands + QD archive + `num_diverse_programs: 2` |
| LLM proposes the same broken change repeatedly | Medium | Artifacts feed stderr back; add the pattern to the system message *without naming targets* |
| Reward hacking | Low–medium | §7.4 guards + manual diff review of top 5 |
| Improvements within noise | **High** | Noise floor from §8.2 in every table; significance tests via Sonia's `validate.py` |
| Contract drift breaks the harness | Low | `allow_full_rewrites: false`; stage-1 arity check |

---

## **13. Deliverables**

**Figure 1 (headline).** Cumulative targets recovered (y, 0–8) vs GPU-hours (x). Three
traces: evolution, random-mutation control, zero-shot control (a horizontal line — it
consumes no GPU time). This is the paper.

**Figure 2.** Best-so-far fitness vs GPU-hours, with horizontal reference lines for the
seed at full training and stock nnU-Net, and a shaded ±2σ noise band.

**Figure 3.** MAP-Elites archive, `params_millions` × `worst_class_dice`, cells coloured
by mean Dice. Shows what quality-diversity retained that pure selection would have lost.

**Table 1.** Per-target: recovered (Y/N), first iteration, GPU-hours to first appearance,
zero-shot control hit rate, random-mutation control hit rate.

**Table 2.** Sealed-test performance: seed, stock nnU-Net, `pengyy` config, top-3 evolved
candidates. Mean Dice, per-class Dice, HD95, with significance tests.

**Table 3.** Off-list discoveries: what evolution found that nnU-Net does not do, and
whether it survived to the final archive.

---

## **14. Order of work**

```
Week 1   Pre-registration committed (§3). Seed program + harness + guards.
         Subsplits written. Zero-shot control run (~1 hour).
Week 1-2 CALIBRATION GATE (§8). Noise floor + Spearman ρ. Go / no-go / escalate.
Week 2   Evaluator complete, stage-1 guards tested against deliberately
         malicious candidates. Dry run at 5 iterations.
Week 2-3 Main evolution, 150 iterations.
Week 3-4 Random-mutation control. Full training of top 3.
Week 4   Sealed test opened once. Target scoring. Figures. Write-up.
```

**Hard dependency:** Sonia's `src/metrics.py` and Caolan's `results/splits.json` must
exist before the calibration gate. Both are week-1 deliverables in Track A; confirm the
`metrics.py` API with Sonia directly rather than waiting.
