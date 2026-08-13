"""Data exploration: headers, label QC, and maturity proxies.

Writes results/fingerprint.json and results/cases.csv. Track B splits import
cases.csv for ICV tertiles. This module does not look at the sealed test
predictions or compute segmentation scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ALL_SUBJECTS,
    CASES_PATH,
    CLASS_NAMES,
    DATA_DIR,
    FINGERPRINT_PATH,
    FOREGROUND_CLASSES,
    IRTK_SUBJECTS,
    MIAL_SUBJECTS,
    RESULTS_DIR,
)


def reconstruction_of(sid: str) -> str:
    return "mial" if sid in MIAL_SUBJECTS else "irtk"


def t2w_path(sid: str) -> Path:
    rec = reconstruction_of(sid)
    return DATA_DIR / f"{sid}_rec-{rec}_T2w.nii.gz"


def dseg_path(sid: str) -> Path:
    rec = reconstruction_of(sid)
    return DATA_DIR / f"{sid}_rec-{rec}_dseg.nii.gz"


def load_labels(path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    nii = nib.load(str(path))
    labels = np.rint(np.asanyarray(nii.dataobj)).astype(np.int64)
    zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])
    return labels, zooms, nii.affine


def surface_voxels(mask: np.ndarray) -> int:
    """Count 6-connected surface voxels of a binary mask."""
    if mask.sum() == 0:
        return 0
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return int(mask.astype(bool).sum() - eroded.sum())


def inspect_subject(sid: str) -> dict:
    rec = reconstruction_of(sid)
    t2_nii = nib.load(str(t2w_path(sid)))
    labels, zooms, affine = load_labels(dseg_path(sid))
    voxel_vol = float(np.prod(zooms))
    unique = sorted(int(v) for v in np.unique(labels))
    if not set(unique) <= set(range(8)):
        raise ValueError(f"{sid}: unexpected labels {unique}")
    missing = sorted(set(FOREGROUND_CLASSES) - set(unique))
    counts = {int(i): int((labels == i).sum()) for i in range(8)}
    icv_voxels = int(sum(counts[i] for i in FOREGROUND_CLASSES))
    icv_mm3 = icv_voxels * voxel_vol
    gm_voxels = counts[2]
    gm_surface = surface_voxels(labels == 2)
    gm_svr = (gm_surface / gm_voxels) if gm_voxels else 0.0
    ventricle_frac = (counts[4] * voxel_vol / icv_mm3) if icv_mm3 else 0.0
    return {
        "subject_id": sid,
        "reconstruction": rec,
        "t2w_shape": list(t2_nii.shape),
        "t2w_dtype": str(t2_nii.get_data_dtype()),
        "t2w_zooms": [float(z) for z in t2_nii.header.get_zooms()[:3]],
        "t2w_affine": np.asarray(t2_nii.affine).tolist(),
        "dseg_shape": list(labels.shape),
        "dseg_zooms": list(zooms),
        "dseg_affine": np.asarray(affine).tolist(),
        "labels_present": unique,
        "missing_classes": missing,
        "voxel_volume_mm3": voxel_vol,
        "icv_voxels": icv_voxels,
        "icv_mm3": icv_mm3,
        "gm_surface_to_volume": gm_svr,
        "ventricle_fraction": ventricle_frac,
        **{f"vol_mm3_class_{i}": counts[i] * voxel_vol for i in range(8)},
        **{f"frac_icv_class_{i}": (counts[i] * voxel_vol / icv_mm3) if icv_mm3 else 0.0 for i in FOREGROUND_CLASSES},
    }


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [inspect_subject(sid) for sid in ALL_SUBJECTS]
    fingerprint = {
        "n_subjects": len(rows),
        "n_mial": sum(1 for r in rows if r["reconstruction"] == "mial"),
        "n_irtk": sum(1 for r in rows if r["reconstruction"] == "irtk"),
        "subjects": [
            {
                "subject_id": r["subject_id"],
                "reconstruction": r["reconstruction"],
                "t2w_shape": r["t2w_shape"],
                "t2w_dtype": r["t2w_dtype"],
                "t2w_zooms": r["t2w_zooms"],
                "t2w_affine": r["t2w_affine"],
                "dseg_shape": r["dseg_shape"],
                "dseg_zooms": r["dseg_zooms"],
                "dseg_affine": r["dseg_affine"],
                "labels_present": r["labels_present"],
                "missing_classes": r["missing_classes"],
            }
            for r in rows
        ],
    }
    FINGERPRINT_PATH.write_text(json.dumps(fingerprint, indent=2))
    fieldnames = [
        "subject_id",
        "reconstruction",
        "icv_mm3",
        "icv_voxels",
        "gm_surface_to_volume",
        "ventricle_fraction",
        "voxel_volume_mm3",
        *[f"vol_mm3_class_{i}" for i in range(8)],
        *[f"frac_icv_class_{i}" for i in FOREGROUND_CLASSES],
        "missing_classes",
    ]
    with CASES_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["missing_classes"] = ",".join(str(c) for c in r["missing_classes"])
            writer.writerow(out)
    print(f"Wrote {FINGERPRINT_PATH} and {CASES_PATH} for {len(rows)} subjects")
    missing = [r["subject_id"] for r in rows if r["missing_classes"]]
    if missing:
        print(f"WARNING: subjects missing a tissue class: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FeTA volumes and write cases.csv")
    parser.parse_args()
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")
    run()


if __name__ == "__main__":
    main()
