# FeTA segmentation — Track B (OpenEvolve)

This repository implements Track B from `OpenEvolve_Spec_FeTA.md`, plus the shared
Track A artifacts it depends on (`src/metrics.py`, `results/splits.json`).

**This Mac is not the scientific host.** Production gates require an RTX A6000
(CUDA, 48 GB). Local `FETA_PROFILE=smoke` runs are for plumbing checks only.

OpenEvolve is pinned to **v0.3.2** (`411fb59c886c18704caaffb611e17cf9e7d824d2`).
Diff-only evolution is `diff_based_evolution: true` (there is no
`allow_full_rewrites` key in 0.3.2). Claude Code does not honor `temperature` or
`max_tokens`; those fields are provenance.

## Layout

```
config.py                 shared paths and class names
src/metrics.py            Dice / HD95 / VS / Euler (imported by both tracks)
src/splits.py             40/20/20 + Track B 10/10
src/validate.py           stats + sealed-test lock
openevolve/               Track B (LLM may edit only initial_program.py evolve block)
  PREREGISTRATION.md      immutable after commit
  settings.py             the only knob file
  generate_config.py      writes config.yaml
  evaluator.py            OpenEvolve contract (isolated subprocess)
  harness/                frozen
```

## Local smoke (this machine)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m openevolve.prompts.lint_system_message || python openevolve/prompts/lint_system_message.py
python src/explore.py
python src/splits.py
python openevolve/generate_config.py
python -c "from harness.guards import write_frozen_hashes; import bootstrap; bootstrap.ensure_paths(); from harness.guards import write_frozen_hashes; write_frozen_hashes()"
# from repo root, with PYTHONPATH=openevolve:.
PYTHONPATH=openevolve:. python -c "from harness.guards import write_frozen_hashes; write_frozen_hashes()"
PYTHONPATH=openevolve:. python openevolve/harness/prepare.py
PYTHONPATH=openevolve:. pytest tests -m "not cuda and not llm and not slow"
```

Reduced MPS loop check:

```bash
FETA_PROFILE=smoke PYTHONPATH=openevolve:. pytest tests/test_seed.py tests/test_guards.py tests/test_metrics.py tests/test_prepare.py tests/test_config_and_targets.py
```

## Prompt leakage gate

`openevolve/prompts/LEAKAGE_REVIEW.md` must contain `SIGNED_OFF=YES` before any
paid LLM call. Automated lint cannot catch paraphrases.

## Production (A6000)

```bash
chmod +x docker/run_a6000.sh
# P0 already done in git.
# P1
PYTHONPATH=openevolve:. python openevolve/harness/prepare.py
# P2
pytest tests/test_seed.py
# P6 seed fitness (60–75 min)
FETA_PROFILE=production PYTHONPATH=openevolve:. python -c "
from evaluator_impl import run_stage2
print(run_stage2('openevolve/initial_program.py'))
"
# P7 zero-shot (costs ~\$5)
python openevolve/controls/zero_shot.py --execute
# P8 calibration (~45 GPU-h)
python openevolve/analysis/calibrate.py
# P9 5-iteration dry run
python openevolve/run_evolution.py --iterations 5
# P10
python openevolve/run_evolution.py --iterations 150
# P11
python openevolve/controls/random_mutation.py --execute --iterations 60
python openevolve/analysis/score_targets.py --checkpoint results/openevolve_output/checkpoints/checkpoint_150
python openevolve/analysis/figures.py
```

Resume:

```bash
python openevolve/run_evolution.py --checkpoint results/openevolve_output/checkpoints/checkpoint_10
```

## Deferred A6000 / paid gates

These are **not** claimed by local smoke:

| Gate | Why deferred here |
| --- | --- |
| Peak VRAM < 14 GB at `[2,1,96,96,96]` | CUDA only |
| 200-step overfit Dice > 0.95 on 96³ | MPS uses 32³ stand-in |
| Seed stage-2 fitness in 60–75 min, Dice > 0.5 | 4000 steps × 4 trains |
| 5-iteration checkpoint dry run with artifacts | LLM cost + GPU hours |
| Zero-shot 20 samples | LLM cost; needs leakage sign-off |
| Spearman ρ calibration | ~40 GPU-h |
| 150-iteration main run | ~140 GPU-h |
| Random-mutation 60 iterations | ~56 GPU-h |
| Sealed test | locked until `results/sealed_test_unlock.json` |

To open the sealed test once:

```json
{"confirm": "OPEN_SEALED_TEST", "reason": "final paper numbers", "candidates": ["..."]}
```

write that to `results/sealed_test_unlock.json`.

## Compatibility patch

`openevolve/patches/parent_lr.py` is version-checked against OpenEvolve 0.3.2 and
writes the parent's `best_lr` beside the temp candidate file. Its SHA-256 is
stamped in `run_manifest.json`.
