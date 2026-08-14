"""src/explore.py: data exploration and QC for the FeTA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    ALL_SUBJECTS,
    CASES_PATH,
    FINGERPRINT_PATH,
    FOREGROUND_CLASSES,
    IRTK_SUBJECTS,
    MIAL_SUBJECTS,
    RESULTS_DIR,
    REPO_ROOT
)

# local raw-data path
FETA_ROOT = REPO_ROOT / "feta_2.4"

PARTICIPANTS_TSV = RESULTS_DIR / "participants.tsv"
QC_FIGURES_DIR = RESULTS_DIR / "qc_figures"
LABEL_ARTIFACTS_DIR = RESULTS_DIR / "label_artifacts"


def subject_files(subject_id: str) -> tuple[Path, Path]:
    """Return (T2w path, dseg path) for a subject under FETA_ROOT."""
    rec = "mial" if subject_id in MIAL_SUBJECTS else "irtk"
    base = FETA_ROOT / subject_id / "anat"
    t2w = base / f"{subject_id}_rec-{rec}_T2w.nii.gz"
    dseg = base / f"{subject_id}_rec-{rec}_dseg.nii.gz"
    return t2w, dseg


def load_header(nifti_path: Path) -> dict:
    """Read shape, voxel spacing, dtype, and affine from a NIfTI file."""
    img = nib.load(nifti_path)
    header = img.header
    return {
        "path": str(nifti_path),
        "shape": [int(x) for x in header.get_data_shape()],
        "spacing": [float(x) for x in header.get_zooms()],
        "dtype": str(header.get_data_dtype()),
        "affine": img.affine.tolist(),
    }


def build_fingerprint(subjects: list[str]) -> list[dict]:
    """Collect header info for T2w and dseg files across all subjects."""
    records = []
    for subject_id in subjects:
        t2w_path, dseg_path = subject_files(subject_id)
        records.append({"subject_id": subject_id, "modality": "T2w", **load_header(t2w_path)})
        records.append({"subject_id": subject_id, "modality": "dseg", **load_header(dseg_path)})
    return records


def validate_labels(dseg_arr: np.ndarray) -> dict:
    """Check that only values 0-7 are present and all foreground classes appear."""
    values_present = set(np.unique(dseg_arr).tolist())
    expected = {0, *FOREGROUND_CLASSES}
    unexpected_values = sorted(values_present - expected)
    missing_classes = sorted(set(FOREGROUND_CLASSES) - values_present)
    return {
        "unexpected_values": unexpected_values,
        "missing_classes": missing_classes,
    }


def compute_volumes(dseg_arr: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """
    Per-class volume, ICV, ventricle fraction, and GM surface-to-volume ratio.

    GM surface-to-volume: boundary-voxel method (count GM voxels with at
    least one non-GM 6-connected neighbour, multiply by mean voxel face
    area). Avoids adding a mesh library (e.g. skimage marching_cubes) as a
    new dependency for the time being.
    """
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    volume_per_class = {}
    for class_idx in FOREGROUND_CLASSES:
        voxel_count = int(np.sum(dseg_arr == class_idx))
        volume_per_class[class_idx] = voxel_count * voxel_volume

    icv_mm3 = sum(volume_per_class.values())
    ventricle_fraction = volume_per_class[4] / icv_mm3 if icv_mm3 > 0 else float("nan")

    gm_mask = dseg_arr == 2
    if gm_mask.any():
        shifted = [
            np.roll(gm_mask, shift, axis=axis)
            for axis in range(3)
            for shift in (1, -1)
        ]
        boundary = gm_mask & ~np.logical_and.reduce([gm_mask == s for s in shifted])
        boundary_voxel_count = int(np.sum(boundary))
        mean_face_area = (
            spacing[0] * spacing[1] + spacing[1] * spacing[2] + spacing[0] * spacing[2]
        ) / 3
        gm_surface_mm2 = boundary_voxel_count * mean_face_area
        gm_surface_to_volume = gm_surface_mm2 / volume_per_class[2]
    else:
        gm_surface_to_volume = float("nan")

    return {
        "icv_mm3": icv_mm3,
        **{f"volume_class_{k}_mm3": v for k, v in volume_per_class.items()},
        "ventricle_fraction": ventricle_fraction,
        "gm_surface_to_volume": gm_surface_to_volume,
    }


def data_quality_check(cases_df: pd.DataFrame) -> dict:
    """
    Spearman correlation between ICV and real gestational age (should be
    strongly positive, since brain volume grows monotonically with age).

    Outlier flag: |z-score of ICV| > 2. Simple default threshold.
    """
    corr, p_value = spearmanr(cases_df["icv_mm3"], cases_df["gestational_age"])

    icv_z = (cases_df["icv_mm3"] - cases_df["icv_mm3"].mean()) / cases_df["icv_mm3"].std()
    outlier_subjects = cases_df.loc[icv_z.abs() > 2, "subject_id"].tolist()

    return {
        "icv_ga_spearman_corr": corr,
        "icv_ga_p_value": p_value,
        "outlier_subjects": outlier_subjects,
    }


def merge_metadata(cases_df: pd.DataFrame) -> pd.DataFrame:
    """Merge gestational age and pathology from PARTICIPANTS_TSV on subject_id."""
    participants = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    participants = participants.rename(columns={
        "participant_id": "subject_id",
        "Gestational age": "gestational_age",
        "Pathology": "pathology",
    })
    return cases_df.merge(participants[["subject_id", "gestational_age", "pathology"]],
                           on="subject_id", how="left")


def qc_slice_figure(t2w_arr: np.ndarray, dseg_arr: np.ndarray, subject_id: str) -> Path:
    """Save the mid-axial T2w slice next to the same slice with label overlay."""
    QC_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mid = t2w_arr.shape[2] // 2

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(t2w_arr[:, :, mid], cmap="gray")
    axes[0].set_title(f"{subject_id} T2w")
    axes[0].axis("off")

    axes[1].imshow(t2w_arr[:, :, mid], cmap="gray")
    axes[1].imshow(dseg_arr[:, :, mid], cmap="tab10", alpha=0.4, vmin=0, vmax=7)
    axes[1].set_title("dseg overlay")
    axes[1].axis("off")

    out_path = QC_FIGURES_DIR / f"{subject_id}_slice.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def label_artifact_figure(dseg_arr: np.ndarray, subject_id: str) -> Path:
    """Save mid-coronal and mid-sagittal views of the label map."""
    LABEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mid_coronal = dseg_arr.shape[1] // 2
    mid_sagittal = dseg_arr.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(dseg_arr[:, mid_coronal, :], cmap="tab10", vmin=0, vmax=7)
    axes[0].set_title(f"{subject_id} coronal")
    axes[0].axis("off")

    axes[1].imshow(dseg_arr[mid_sagittal, :, :], cmap="tab10", vmin=0, vmax=7)
    axes[1].set_title("sagittal")
    axes[1].axis("off")

    out_path = LABEL_ARTIFACTS_DIR / f"{subject_id}_artifacts.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FeTA data exploration and QC")
    parser.add_argument("--subjects", nargs="*", default=None,
                         help="Subset of subject IDs for a quick test run, e.g. sub-001 sub-002")
    parser.add_argument("--skip-figures", action="store_true",
                         help="Skip figure generation")
    args = parser.parse_args()
    subjects = args.subjects or list(ALL_SUBJECTS)

    fingerprint = build_fingerprint(subjects)
    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINGERPRINT_PATH, "w") as f:
        json.dump(fingerprint, f, indent=2)

    rows = []
    loaded = {}  # cache arrays for the figure pass below
    for subject_id in subjects:
        t2w_path, dseg_path = subject_files(subject_id)
        t2w_arr = nib.load(t2w_path).get_fdata()
        dseg_img = nib.load(dseg_path)
        dseg_arr = np.round(dseg_img.get_fdata()).astype(int)
        spacing = dseg_img.header.get_zooms()

        label_report = validate_labels(dseg_arr)
        volumes = compute_volumes(dseg_arr, spacing)
        rows.append({"subject_id": subject_id, **volumes, **label_report})
        loaded[subject_id] = (t2w_arr, dseg_arr)

    cases_df = pd.DataFrame(rows)
    cases_df = merge_metadata(cases_df)

    dq_report = data_quality_check(cases_df)
    print("Data quality check:", dq_report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases_df.to_csv(CASES_PATH, index=False)

    if not args.skip_figures:
        for subject_id in subjects:
            t2w_arr, dseg_arr = loaded[subject_id]
            qc_slice_figure(t2w_arr, dseg_arr, subject_id)
            label_artifact_figure(dseg_arr, subject_id)

    print(f"Done. Wrote {FINGERPRINT_PATH} and {CASES_PATH}")


if __name__ == "__main__":
    main()