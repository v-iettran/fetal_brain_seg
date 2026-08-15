# OpenEvolve remote A6000 runbook

This runbook assumes the repository and FeTA data have been copied to `/feta`
on an NVIDIA A6000 server. Run every command from `/feta`. The scientific
profile is CUDA-only; results from the local `smoke` profile are not comparable.

## 1. One-time server setup

Start a persistent terminal because the experiment takes several days:

```bash
cd /feta
tmux new -s feta
```

Create an isolated Python environment and install the pinned runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export FETA_PROFILE=production
export PYTHONPATH=openevolve:.
mkdir -p logs
```

The shell running evolution must retain those two exported variables. Verify
the framework, CUDA, and Claude Code CLI:

```bash
python - <<'PY'
from importlib.metadata import version
import torch

print("OpenEvolve:", version("openevolve"))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert version("openevolve") == "0.3.2"
assert torch.cuda.is_available(), "production requires CUDA"
PY

nvidia-smi
claude --version
```

Authenticate Claude Code interactively if the server has not already been
authenticated. Do not start a paid control or evolution run until this works.

## 2. Data and immutable-input checks

Raw files must be under `/feta/mri_gz` using names such as
`sub-001_rec-mial_T2w.nii.gz`. There must be 80 images and 80 segmentations.

If `/feta/cache/npy/index.json` was copied from the local machine, reuse it.
Otherwise create the cache once:

```bash
PYTHONPATH=openevolve:. python openevolve/harness/prepare.py \
  2>&1 | tee logs/prepare.log
```

Generate and validate the production YAML, but do **not** regenerate frozen
hashes:

```bash
python openevolve/generate_config.py
python openevolve/prompts/lint_system_message.py

python - <<'PY'
import bootstrap
bootstrap.ensure_paths()
from harness.guards import verify_frozen_files
verify_frozen_files()
print("Frozen hashes: OK")
PY
```

The working tree should be clean before a scientific run:

```bash
git status --short
```

## 3. Required human leakage review

A person who has not read the T1–T8 target list must review:

- `openevolve/prompts/system_message.txt`
- `openevolve/prompts/LEAKAGE_REVIEW.md`

That reviewer checks every box and sets:

```text
SIGNED_OFF=YES
REVIEWER=<name>
DATE=<YYYY-MM-DD>
NOTES=<optional>
```

Commit the completed review before the first paid call. Never use
`--skip-review-gate` for a scientific run.

## 4. Hardware and seed gates

Run the non-paid tests first:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest \
  -m "not llm and not slow" 2>&1 | tee logs/tests.log
```

Check isolated Stage 1, including model shape, gradients, parameter count, and
the 14 GB peak-VRAM gate:

```bash
python - <<'PY' 2>&1 | tee logs/seed_stage1.log
from evaluator_impl import run_stage1
result = run_stage1("openevolve/initial_program.py")
print(result)
assert result["ok"], result
PY
```

Then run the production seed fitness gate. This performs the complete Optuna,
training, inference, and shared-metric path and is expected to take roughly
60–75 minutes:

```bash
python - <<'PY' 2>&1 | tee logs/seed_stage2.log
from evaluator_impl import run_stage2
result = run_stage2("openevolve/initial_program.py")
print(result)
assert result["ok"], result
assert result["combined_score"] > 0.5, result
PY
```

Stop and diagnose the run if either seed gate fails.

## 5. Zero-shot control

Run this before evolution. It makes 20 independent Claude calls without
fitness feedback:

```bash
python openevolve/controls/zero_shot.py \
  --execute --n 20 --out results/zero_shot \
  2>&1 | tee logs/zero_shot.log
```

Expected outputs:

- `results/zero_shot/index.json`
- `results/zero_shot/sample_*.py`
- `results/zero_shot_targets.json`

Back these files up before continuing.

## 6. Proxy calibration gate

The required Spearman gate needs a JSON list of eight full-training scores, in
this exact order:

```text
seed, instance_norm, dice_term, deep_supervision,
lr_10x, base8, sgd_poly, foreground_sampling
```

After those scores have been produced by the separate full-training pipeline,
save them as `results/calibration_full_scores.json`. Then run the five-repeat
noise floor, all eight proxy variants, and the rank decision in one command:

```bash
python openevolve/analysis/calibrate.py \
  --full-scores results/calibration_full_scores.json \
  --out results/calibration.json \
  2>&1 | tee logs/calibration_gate.log
```

Proceed only when `decision` is `proceed` or `proceed_with_limitation`. If it
is `stop_escalate`, stop; the pre-registered escalation requires a dated
amendment and must not be applied automatically.

## 7. Five-iteration dry run

Use a separate output directory so the dry run cannot contaminate the main
population:

```bash
python openevolve/run_evolution.py \
  --iterations 5 \
  --output results/openevolve_dry_run \
  2>&1 | tee logs/evolution_dry_run.log
```

Confirm that the output contains checkpoints, programs, archive metadata, and
evaluation artifacts. Do not reuse this checkpoint for the main run.

## 8. Full 150-iteration experiment

Start the preregistered main search in its own output directory:

```bash
python openevolve/run_evolution.py \
  --iterations 150 \
  --output results/openevolve_output \
  2>&1 | tee logs/evolution_main.log
```

Detach from tmux with `Ctrl-b`, then `d`. Reconnect with:

```bash
tmux attach -t feta
```

If the run is interrupted, resume from the newest complete checkpoint:

```bash
python openevolve/run_evolution.py \
  --checkpoint results/openevolve_output/checkpoints/checkpoint_<N> \
  --output results/openevolve_output \
  2>&1 | tee -a logs/evolution_main.log
```

Do not change `settings.py`, the split JSON files, metric code, prompts, or
frozen hashes after iteration 1.

## 9. Fitness-blind random-mutation control

Run 60 iterations with the same production profile and evaluation budget:

```bash
python openevolve/controls/random_mutation.py \
  --execute --iterations 60 --out results/random_mutation \
  2>&1 | tee logs/random_mutation.log
```

This control still calls Claude and evaluates every syntactically valid child,
but fitness is redacted from future parent selection.

## 10. Score target recovery and generate artifacts

After the main checkpoint is complete:

```bash
python openevolve/analysis/score_targets.py \
  --checkpoint results/openevolve_output/checkpoints/checkpoint_150 \
  --out results/target_scores.json \
  2>&1 | tee logs/score_targets.log

python openevolve/analysis/figures.py \
  --results-dir results \
  2>&1 | tee logs/figures.log
```

Copy `results/`, `logs/`, the completed prompt review, the exact git commit
hash, and `openevolve/config.yaml` to durable storage.

## 11. Work that is intentionally not automatic

The proxy calibration requires eight full-training scores (1000-epoch
equivalent, five-fold) supplied to `calibrate.py`. Final five-fold training of
the top three evolved candidates, the seed, and controls is also separate from
the evolutionary proxy harness.

The sealed test must remain unopened until the final candidates are fixed.
Do not create `results/sealed_test_unlock.json` during controls, calibration,
the dry run, or the 150-iteration search.
