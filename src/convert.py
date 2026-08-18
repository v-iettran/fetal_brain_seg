"""Convert the training-split subjects into nnU-Net's expected raw layout.

Reads results/splits.json (never regenerates it), copies each training
subject's image + label into nnUNet_raw/<DATASET_NAME>/{imagesTr,labelsTr},
and writes dataset.json. Only the 40 train-split subjects are touched --
tuning and test subjects must never enter nnU-Net's raw/preprocessed folders.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CLASS_NAMES,
    DATA_DIR,
    IRTK_SUBJECTS,
    MIAL_SUBJECTS,
    NNUNET_DATASET_NAME,
    NNUNET_RAW_DIR,
    SPLITS_PATH,
)


def _reconstruction_for(subject_id: str) -> str:
    if subject_id in MIAL_SUBJECTS:
        return "mial"
    if subject_id in IRTK_SUBJECTS:
        return "irtk"
    raise ValueError(f"Unknown subject: {subject_id}")


def _load_train_ids() -> list[str]:
    if not SPLITS_PATH.exists():
        raise SystemExit(
            f"{SPLITS_PATH} not found. Run `python src/splits.py` first "
            "(or pull it -- it's already committed)."
        )
    payload = json.loads(SPLITS_PATH.read_text())
    return sorted(payload["train"])


def _convert_one(subject_id: str, dataset_dir: Path) -> None:
    recon = _reconstruction_for(subject_id)
    img_src = DATA_DIR / f"{subject_id}_rec-{recon}_T2w.nii.gz"
    lbl_src = DATA_DIR / f"{subject_id}_rec-{recon}_dseg.nii.gz"
    if not img_src.exists() or not lbl_src.exists():
        raise FileNotFoundError(f"Missing source files for {subject_id}: {img_src}, {lbl_src}")

    img_dst = dataset_dir / "imagesTr" / f"{subject_id}_0000.nii.gz"
    lbl_dst = dataset_dir / "labelsTr" / f"{subject_id}.nii.gz"

    # Image copies through unchanged -- nnU-Net's own preprocessing handles
    # resampling/normalisation itself (see README: "not writing a preprocessing
    # pipeline").
    shutil.copyfile(img_src, img_dst)

    # Labels are stored float32 in the source data; nnU-Net requires integer
    # label maps. Round (not truncate) before casting, and verify the result
    # only contains our known classes.
    lbl_img = nib.load(lbl_src)
    lbl_data = np.asarray(lbl_img.dataobj)
    rounded = np.rint(lbl_data).astype(np.uint8)
    bad_values = set(np.unique(rounded)) - set(CLASS_NAMES)
    if bad_values:
        raise ValueError(f"{subject_id}: label values outside 0-7 after rounding: {bad_values}")
    out_img = nib.Nifti1Image(rounded, lbl_img.affine, lbl_img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, lbl_dst)


def _write_dataset_json(dataset_dir: Path, n_training: int) -> None:
    labels = {name: label_id for label_id, name in CLASS_NAMES.items()}
    payload = {
        "channel_names": {"0": "T2w"},
        "labels": labels,
        "numTraining": n_training,
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")


def convert(force: bool = False) -> Path:
    train_ids = _load_train_ids()
    if len(train_ids) != 40:
        raise ValueError(f"Expected 40 train subjects, got {len(train_ids)}")

    dataset_dir = NNUNET_RAW_DIR / NNUNET_DATASET_NAME
    if dataset_dir.exists() and not force:
        raise SystemExit(f"{dataset_dir} already exists. Pass --force to overwrite.")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    for subject_id in train_ids:
        _convert_one(subject_id, dataset_dir)

    _write_dataset_json(dataset_dir, len(train_ids))

    print(f"Wrote {len(train_ids)} subjects to {dataset_dir}")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert train split to nnU-Net raw format")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing dataset dir")
    args = parser.parse_args()
    convert(force=args.force)


if __name__ == "__main__":
    main()