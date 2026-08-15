# OpenEvolve Track B — James handoff and decision checklist

This document is the quick onboarding guide for James. It summarises what is
implemented, what is already fixed, and what Viet and James must agree before
the first scientific OpenEvolve run.

Repository: <https://github.com/v-iettran/fetal_brain_seg>

## 1. Research question

Starting from the original Çiçek 2016 3D U-Net, can LLM-guided evolutionary
code search rediscover design decisions that distinguish it from nnU-Net?

The claim is **rediscovery as a function of compute**, not “beat nnU-Net”.
Recovery is assessed against eight pre-registered targets, with zero-shot and
fitness-blind controls.

## 2. What is implemented

### Evolvable program

`initial_program.py` contains the faithful ~19.1M-parameter 3D U-Net seed and
the six-function contract:

1. `build_model`
2. `build_loss`
3. `build_optimizer`
4. `build_scheduler`
5. `build_sampler`
6. `build_augmentation`

Only code between the `EVOLVE-BLOCK` markers may change.

### Frozen evaluation side

- `harness/prepare.py`: 0.5 mm resampling, image-derived crop, padding and
  non-zero z-score normalisation.
- `harness/data.py`: deterministic patch batches and candidate-controlled
  sampler/augmentation checks.
- `harness/train_loop.py`: AMP, gradient clipping, deterministic seeds and NaN
  abort.
- `harness/infer.py`: Gaussian-weighted sliding-window inference.
- `harness/guards.py`: frozen-file hashes, import restrictions, contract checks,
  split integrity and prediction sanity checks.
- `candidate_runner.py` / `candidate_worker.py`: candidate execution in a
  killable subprocess with scrubbed LLM credentials.

### Fitness and search

- `evaluator.py`: OpenEvolve-facing cascade wrapper.
- `evaluator_impl.py`: stage-1 structural checks and stage-2 Optuna/training
  fitness.
- `settings.py`: the only human-edited knob file.
- `generate_config.py`: generates `config.yaml`; the YAML is not hand-edited.
- OpenEvolve is pinned to **0.3.2**.
- `patches/parent_lr.py` passes the parent’s best learning rate into the child’s
  Optuna study.

### Controls and analysis

- `controls/zero_shot.py`: 20 independent no-fitness LLM samples.
- `controls/random_mutation.py`: uniform parents/inspirations, fitness hidden
  from selection, every valid child retained.
- `analysis/calibrate.py`: five-seed noise floor and proxy/full-ranking gate.
- `analysis/score_targets.py`: T1–T8 static/runtime detection.
- `analysis/figures.py`: required figures and result tables from JSON.

## 3. Current verification status

Completed locally:

- all 80 MRI cases pass preprocessing and crop/pad round-trip checks;
- deterministic 40/20/20 and Track B 10/10 splits were generated locally;
- 27 non-GPU unit tests pass;
- reduced 32³ MPS single-patch overfit test passes;
- system-message direct-keyword leakage lint passes;
- generated configuration loads under OpenEvolve 0.3.2;
- settings validation rejects invalid patch sizes.

Not yet run:

- exact CUDA batch-2 96³ forward/backward and `<14 GB` VRAM gate;
- production seed stage-2 fitness;
- zero-shot control;
- calibration and Spearman gate;
- five-iteration OpenEvolve dry run;
- 150-iteration main run;
- random-mutation control;
- sealed-test evaluation.

## 4. Decisions that are already fixed

These are in `PREREGISTRATION.md`. Changing them requires a dated amendment;
changing experiment settings after iteration 1 invalidates the run.

- Seed: Çiçek architecture with deliberately minimal training recipe.
- Main run: 150 iterations, 3 islands, migration every 25.
- Diff-based evolution; full rewrites disabled.
- Patch 96³, batch size 2, 4,000 proxy steps.
- Inner tuning: learning rate only, 3 trials plus fresh re-evaluation.
- Fitness: mean Dice over classes 1–7 on a separate fitness split.
- Archive axes: parameter count × worst-class Dice, 6×6 cells.
- Controls: 20-sample zero-shot and 60-iteration random mutation.
- Success: at least 4/8 recovered and recovery exceeds both controls.
- The sealed test is opened once for at most three final candidates.
- Track B uses the team-owned `src/metrics.py::compute_metrics` through a
  frozen adapter, so evolved and manually implemented models are scored by the
  same code. The split implementation remains self-contained.

## 5. Decisions Viet and James need to agree on

Record the outcome of each item before any paid or production run.

### D1 — James’s review role

Choose one:

- **Blind prompt reviewer:** James first reads only
  `prompts/system_message.txt` and `prompts/LEAKAGE_REVIEW.md`, signs the review,
  and only then reads the target list.
- **Full co-owner:** James reads `PREREGISTRATION.md` immediately; a different
  person who has not read T1–T8 must perform the leakage review.

Current status: **undecided**.

### D2 — LLM ensemble and spending limits

Current proposal in `settings.py`:

- Sonnet: weight 0.8, per-call budget cap $40.
- Opus: weight 0.2, per-call budget cap $30.
- 3 top programs + 2 diverse programs in each prompt.

Agree on:

- whether to keep the 80/20 ensemble;
- whether the per-call budget caps are acceptable;
- whether the same ensemble is used for zero-shot and evolution.

Important: OpenEvolve’s Claude Code backend currently does not enforce the
configured temperature or max-token values. They are provenance fields only.

### D3 — Search breadth

Current proposal:

- population 120;
- 3 islands;
- migration every 25 iterations;
- archive size 36;
- 6×6 `params_millions × worst_class_dice` grid;
- 3 concurrent evaluations.

Agree whether this provides enough diversity for 150 iterations without
spreading evaluations too thinly.

### D4 — Proxy fitness budget

Current proposal:

- 40 training cases;
- 96³ patches, batch 2;
- 4,000 optimiser steps;
- 3 Optuna LR trials;
- one fresh-seed fitness re-evaluation;
- LR range `1e-4` to `1e-1`.

Agree that the compute/noise trade-off is acceptable. The calibration gate,
not preference, decides whether this proxy is retained:

- `ρ ≥ 0.7`: proceed;
- `0.6 ≤ ρ < 0.7`: proceed and report the limitation;
- `ρ < 0.6`: stop and amend the protocol before continuing.

### D5 — Stage-1 rejection policy

Current proposal:

- parameter count between 1M and 150M;
- output and loss must be finite;
- at least 90% of trainable parameters have finite gradients;
- peak CUDA memory below 14 GB;
- cascade threshold 0.35;
- evaluator timeout 7,200 seconds;
- augmentation timeout 2 seconds per patch.

Agree whether these rules reject only broken/reward-hacking candidates rather
than valid unconventional designs.

### D6 — Control fidelity

Review both control implementations and agree that:

- zero-shot gets the seed and the registered generic prompt, but no fitness or
  feedback;
- random mutation uses the same generation/diff machinery;
- random parents and inspirations are uniform;
- performance metrics cannot affect future parent selection;
- every syntactically valid child is retained;
- recovery is compared per iteration.

### D7 — Target detectors

Review `analysis/score_targets.py`, especially:

- T4: model returns multiple outputs and the loss genuinely consumes at least
  two;
- T6: SGD/Nesterov plus monotonic schedule below 10% of initial LR;
- T7: empirical forced-foreground probability is at least 0.2;
- T8: at least three additional augmentation families.

The detectors should be conservative: a false positive weakens the paper more
than a missed ambiguous candidate.

### D8 — Manual review and stop/go rules

Agree that:

- top-five candidate diffs are inspected manually;
- the main run starts only after zero-shot and calibration are recorded;
- failed calibration stops the run;
- no settings change is made during a run;
- the sealed test remains inaccessible until candidates are frozen.

## 6. Recommended review order

### If James is the blind prompt reviewer

1. Read `prompts/system_message.txt`.
2. Complete `prompts/LEAKAGE_REVIEW.md`.
3. Read this document.
4. Read `PREREGISTRATION.md`.
5. Review D2–D8 and `settings.py`.
6. Review the controls, evaluator and target detectors.

### If James is a full co-owner

1. Read this document.
2. Read `PREREGISTRATION.md`.
3. Review D2–D8 and `settings.py`.
4. Review the controls, evaluator and target detectors.
5. Find a separate blind reviewer for the system message.

## 7. Quick local orientation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python openevolve/generate_config.py
PYTHONPATH=openevolve:. pytest tests -m "not cuda and not llm and not slow"
```

Do not execute `zero_shot.py --execute` or start evolution until D1–D8 are
recorded and the leakage review is signed.

## 8. Decision record

| Decision | Agreed value | Viet | James | Date |
| --- | --- | --- | --- | --- |
| D1 Review role |  |  |  |  |
| D2 LLM ensemble/budget |  |  |  |  |
| D3 Search breadth |  |  |  |  |
| D4 Proxy budget |  |  |  |  |
| D5 Stage-1 rejection |  |  |  |  |
| D6 Control fidelity |  |  |  |  |
| D7 Target detectors |  |  |  |  |
| D8 Manual gates |  |  |  |  |
