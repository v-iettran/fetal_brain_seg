"""Run a trained nnU-Net model on the tuning and test splits.

Writes predicted label maps to results/predictions/<config_name>/ for Sonia
(validate.py) and Albee (interpret.py) to consume. Test-split predictions
must never be looked at/scored until the sealed test set is formally opened
(see README: results/sealed_test_unlock.json).

NOTE: draft, written without a working nnunetv2 install to check against.
The nnUNetv2_predict Python API surface below (predict_from_files) is the
current-ish v2 entrypoint but should be diffed against whatever nnunetv2
version ends up pinned, first time this runs for real.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DATA_DIR,
    IRTK_SUBJECTS,
    MIAL_SUBJECTS,
    NNUNET_DATASET_NAME,
    NNUNET_RAW_DIR,
    RESULTS_DIR,
    SPLITS_PATH,
)

PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def _reconstruction_for(subject_id: str) -> str:
    if subject_id in MIAL_SUBJECTS:
        return "mial"
    if subject_id in IRTK_SUBJECTS:
        return "irtk"
    raise ValueError(f"Unknown subject: {subject_id}")


def _load_split_ids(split_name: str) -> list[str]:
    if split_name == "test":
        unlock_path = RESULTS_DIR / "sealed_test_unlock.json"
        if not unlock_path.exists():
            raise SystemExit(
                "results/sealed_test_unlock.json not present -- the sealed test "
                "set is locked. Do not predict on it until the team formally "
                "opens it (see README)."
            )
    payload = json.loads(SPLITS_PATH.read_text())
    if split_name not in payload:
        raise ValueError(f"Unknown split '{split_name}', expected train/tuning/test")
    return sorted(payload[split_name])


def _stage_input_folder(subject_ids: list[str], staging_dir: Path) -> None:
    """nnUNetv2_predict wants a flat folder of <case>_0000.nii.gz inputs."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    for subject_id in subject_ids:
        recon = _reconstruction_for(subject_id)
        src = DATA_DIR / f"{subject_id}_rec-{recon}_T2w.nii.gz"
        if not src.exists():
            raise FileNotFoundError(f"Missing source image for {subject_id}: {src}")
        dst = staging_dir / f"{subject_id}_0000.nii.gz"
        shutil.copyfile(src, dst)


def predict(
    split_name: str,
    config_name: str,
    fold: str = "all",
    trainer_class_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
) -> Path:
    """Run inference for one config on one split, writing to
    results/predictions/<config_name>/<split_name>/.

    `fold="all"` uses nnU-Net's ensembled 5-fold prediction, matching how the
    challenge method's "ensemble learning" is described in the pipeline doc.
    """
    subject_ids = _load_split_ids(split_name)

    staging_dir = RESULTS_DIR / "_predict_staging" / split_name
    _stage_input_folder(subject_ids, staging_dir)

    output_dir = PREDICTIONS_DIR / config_name / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deferred import: nnunetv2 isn't a hard dependency of every script in
    # this repo (e.g. splits.py doesn't need it), so keep the import local
    # to the function that actually needs it.
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(
            Path.home()
            / "nnUNet_results"
            / NNUNET_DATASET_NAME
            / f"{trainer_class_name}__{plans_identifier}__{config_name}"
        ),
        use_folds=(fold,) if fold != "all" else None,
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        str(staging_dir),
        str(output_dir),
        save_probabilities=False,
        overwrite=True,
    )

    print(f"Wrote {len(subject_ids)} predictions to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a trained nnU-Net model")
    parser.add_argument("--split", choices=["tuning", "test"], required=True)
    parser.add_argument(
        "--config",
        choices=["vanilla_unet", "nnunet_3d_fullres", "pengyy_48feat"],
        required=True,
        help="Which of the three trained configurations to predict with",
    )
    parser.add_argument("--fold", default="all", help="Fold to use, or 'all' for the 5-fold ensemble")
    args = parser.parse_args()
    predict(split_name=args.split, config_name=args.config, fold=args.fold)


if __name__ == "__main__":
    main()