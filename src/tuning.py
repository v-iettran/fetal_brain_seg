"""Optuna hyperparameter search on the tuning split.

Trains short (250-epoch) runs on the 40 train-split subjects, evaluates on
the 20 tuning-split subjects using Sonia's metrics.py, and searches:
  - learning rate (log, 1e-3 to 1e-1)
  - weight decay (log, 1e-6 to 1e-3)
  - optimizer (SGD+Nesterov / Adam / AdamW)
  - Dice:CE loss weight ratio (0.25 to 4.0)
  - foreground oversampling fraction (0.2 to 0.6)

IMPORTANT -- report the retrained model, not the best trial (see pipeline
doc): after study.best_params is known, retrain from scratch at the full
1000 epochs and report THAT score, not the shortened-schedule trial score.
retrain_best() below does that step; it is a separate call, not automatic,
so nobody accidentally reports the inflated number.

NOTE: draft. The train-for-N-epochs / evaluate loop below is a thin sketch
around nnU-Net's own dataloaders -- it has not been run, since this machine
has no CUDA and no nnU-Net install. Treat _train_and_evaluate as the part
most likely to need real rework once someone can actually execute it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NNUNET_DATASET_NAME, RESULTS_DIR, SEED, SPLITS_PATH  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402  -- Sonia's module; pure functions, (pred, gt, spacing) -> dict

STUDY_DB_PATH = RESULTS_DIR / "optuna_study.db"
TRIAL_RESULTS_PATH = RESULTS_DIR / "trackA" / "tuning_trials.json"
SHORT_SCHEDULE_EPOCHS = 250
FULL_SCHEDULE_EPOCHS = 1000
N_TRIALS = 30


def _load_tuning_ids() -> list[str]:
    payload = json.loads(SPLITS_PATH.read_text())
    return sorted(payload["tuning"])


def _sample_hyperparams(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd_nesterov", "adam", "adamw"]),
        "dice_ce_ratio": trial.suggest_float("dice_ce_ratio", 0.25, 4.0),
        "fg_oversample_fraction": trial.suggest_float("fg_oversample_fraction", 0.2, 0.6),
    }


def _train_and_evaluate(
    hyperparams: dict,
    num_epochs: int,
    tuning_ids: list[str],
    trial=None,
) -> float:
    """Train with the given hyperparams for num_epochs, evaluate mean Dice
    across classes 1-7 on the tuning subjects. Reports intermediate values to
    `trial` (if given) so Optuna's MedianPruner can kill bad trials early.

    Placeholder structure -- the actual nnU-Net training-step wiring
    (dataloader, network, optimizer construction from `hyperparams`,
    checkpointing) needs writing against a real nnU-Net install. Left as a
    stub with the correct shape (inputs, epoch loop, pruning hook, metrics
    call) rather than guessed internals that would likely be wrong.
    """
    raise NotImplementedError(
        "Training loop body not yet written -- needs a real nnunetv2 install "
        "to build correctly against its dataloader/network APIs. Structure "
        "(epoch loop + trial.report + compute_metrics call) is sketched "
        "below in the docstring for whoever picks this up:\n\n"
        "for epoch in range(num_epochs):\n"
        "    train_one_epoch(...)\n"
        "    if epoch % 25 == 0:\n"
        "        dice = evaluate_on(tuning_ids, compute_metrics)\n"
        "        if trial is not None:\n"
        "            trial.report(dice, epoch)\n"
        "            if trial.should_prune():\n"
        "                raise optuna.TrialPruned()\n"
        "return final_dice\n"
    )


def objective(trial, tuning_ids: list[str]) -> float:
    hyperparams = _sample_hyperparams(trial)
    dice = _train_and_evaluate(hyperparams, SHORT_SCHEDULE_EPOCHS, tuning_ids, trial=trial)
    return dice


def run_search(n_trials: int = N_TRIALS) -> "optuna.Study":
    import optuna

    tuning_ids = _load_tuning_ids()
    RESULTS_DIR.joinpath("trackA").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
        storage=f"sqlite:///{STUDY_DB_PATH}",
        study_name=NNUNET_DATASET_NAME,
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, tuning_ids), n_trials=n_trials)

    TRIAL_RESULTS_PATH.write_text(
        json.dumps(
            {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "short_schedule_epochs": SHORT_SCHEDULE_EPOCHS,
                "note": "Score above is on the SHORT schedule -- not the reportable number. "
                "Call retrain_best() for the number that goes in the paper.",
            },
            indent=2,
        )
    )
    print(f"Best params: {study.best_params} (short-schedule Dice: {study.best_value:.4f})")
    return study


def retrain_best(study: "optuna.Study") -> float:
    """Retrain the winning hyperparams from scratch at the full schedule and
    return that Dice score -- this is the number to report, per the pipeline
    doc's explicit warning against reporting the best trial's shortened-
    schedule score (which inflates by ~1 noise-floor stddev).
    """
    tuning_ids = _load_tuning_ids()
    dice = _train_and_evaluate(study.best_params, FULL_SCHEDULE_EPOCHS, tuning_ids, trial=None)
    print(f"Full-schedule retrain Dice: {dice:.4f}")
    return dice


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument(
        "--retrain-best",
        action="store_true",
        help="After the search, retrain the winner at the full 1000-epoch schedule",
    )
    args = parser.parse_args()
    study = run_search(n_trials=args.n_trials)
    if args.retrain_best:
        retrain_best(study)