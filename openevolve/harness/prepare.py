"""One-off preprocessing: NIfTI -> .npy cache. Frozen, outside the search space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALL_SUBJECTS, CACHE_DIR, DATA_DIR, IRTK_SUBJECTS, MIAL_SUBJECTS  # noqa: E402

TARGET_SPACING = 0.5
MARGIN = 16
MIN_AXIS = 96
DIVISOR = 8


def reconstruction_of(sid: str) -> str:
    return "mial" if sid in MIAL_SUBJECTS else "irtk"


def _paths(sid: str) -> tuple[Path, Path]:
    rec = reconstruction_of(sid)
    return (
        DATA_DIR / f"{sid}_rec-{rec}_T2w.nii.gz",
        DATA_DIR / f"{sid}_rec-{rec}_dseg.nii.gz",
    )


def resample_isotropic(volume: np.ndarray, spacing: tuple[float, ...], order: int) -> np.ndarray:
    factors = tuple(float(s) / TARGET_SPACING for s in spacing)
    return zoom(volume, factors, order=order, mode="nearest", prefilter=order > 0)


def crop_to_brain(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
    nonzero = np.argwhere(image != 0)
    if nonzero.size == 0:
        raise ValueError("image is entirely zero; cannot crop")
    mins = np.maximum(nonzero.min(axis=0) - MARGIN, 0)
    maxs = np.minimum(nonzero.max(axis=0) + MARGIN + 1, image.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    bbox = (int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(mins[2]), int(maxs[2]))
    return image[sl], bbox


def apply_bbox(volume: np.ndarray, bbox: tuple[int, int, int, int, int, int]) -> np.ndarray:
    d0, d1, h0, h1, w0, w1 = bbox
    return volume[d0:d1, h0:h1, w0:w1]


def pad_to_constraints(volume: np.ndarray) -> tuple[np.ndarray, list[list[int]], tuple[int, ...]]:
    pre_pad = tuple(int(s) for s in volume.shape)
    target = []
    pads = []
    for s in pre_pad:
        t = max(MIN_AXIS, s)
        t = ((t + DIVISOR - 1) // DIVISOR) * DIVISOR
        target.append(t)
        extra = t - s
        before = extra // 2
        after = extra - before
        pads.append([before, after])
    padded = np.pad(volume, pads, mode="constant", constant_values=0)
    return padded, pads, pre_pad


def unpad(volume: np.ndarray, pads: list[list[int]], pre_pad: tuple[int, ...]) -> np.ndarray:
    sl = tuple(slice(b, b + s) for (b, _), s in zip(pads, pre_pad))
    out = volume[sl]
    if out.shape != pre_pad:
        raise ValueError(f"unpad shape {out.shape} != {pre_pad}")
    return out


def restore_to_resampled(
    cropped_or_padded: np.ndarray,
    bbox: tuple[int, int, int, int, int, int],
    resampled_shape: tuple[int, ...],
    pads: list[list[int]] | None = None,
    pre_pad: tuple[int, ...] | None = None,
) -> np.ndarray:
    vol = cropped_or_padded
    if pads is not None and pre_pad is not None:
        vol = unpad(vol, pads, pre_pad)
    restored = np.zeros(resampled_shape, dtype=vol.dtype)
    d0, d1, h0, h1, w0, w1 = bbox
    restored[d0:d1, h0:h1, w0:w1] = vol
    return restored


def zscore_nonzero(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    mask = image != 0
    if mask.sum() == 0:
        raise ValueError("no non-zero voxels for z-score")
    mean = float(image[mask].mean())
    std = float(image[mask].std())
    if std < 1e-8:
        std = 1.0
    out = np.zeros_like(image, dtype=np.float32)
    out[mask] = ((image[mask] - mean) / std).astype(np.float32)
    return out, mean, std


def process_subject(sid: str, cache_dir: Path) -> dict:
    t2_path, dseg_path = _paths(sid)
    t2_nii = nib.load(str(t2_path))
    dseg_nii = nib.load(str(dseg_path))
    image = np.asanyarray(t2_nii.dataobj).astype(np.float32)
    labels = np.rint(np.asanyarray(dseg_nii.dataobj)).astype(np.int64)
    unique = set(int(v) for v in np.unique(labels))
    if not unique <= set(range(8)):
        raise ValueError(f"{sid}: labels {sorted(unique)} not in 0..7")
    spacing = tuple(float(z) for z in t2_nii.header.get_zooms()[:3])
    image_r = resample_isotropic(image, spacing, order=1).astype(np.float32)
    labels_r = np.rint(resample_isotropic(labels.astype(np.float32), spacing, order=0)).astype(np.int64)
    cropped_img, bbox = crop_to_brain(image_r)
    cropped_lbl = apply_bbox(labels_r, bbox)
    padded_img, pads, pre_pad = pad_to_constraints(cropped_img)
    padded_lbl, pads_l, pre_pad_l = pad_to_constraints(cropped_lbl)
    if pads != pads_l or pre_pad != pre_pad_l:
        raise RuntimeError(f"{sid}: image/label pad mismatch")
    # Round-trip crop/pad on labels before z-score (z-score is not invertible).
    restored = restore_to_resampled(padded_lbl, bbox, labels_r.shape, pads, pre_pad)
    if not np.array_equal(restored, labels_r):
        raise AssertionError(f"{sid}: crop/pad round-trip failed")
    normed, mean, std = zscore_nonzero(padded_img)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{sid}_img.npy", np.ascontiguousarray(normed, dtype=np.float32))
    np.save(cache_dir / f"{sid}_lbl.npy", np.ascontiguousarray(padded_lbl, dtype=np.int64))
    meta = {
        "subject_id": sid,
        "reconstruction": reconstruction_of(sid),
        "original_spacing": list(spacing),
        "original_shape": list(image.shape),
        "original_affine": np.asarray(t2_nii.affine).tolist(),
        "resampled_shape": list(image_r.shape),
        "target_spacing": TARGET_SPACING,
        "bbox": list(bbox),
        "pads": pads,
        "pre_pad_shape": list(pre_pad),
        "cached_shape": list(normed.shape),
        "zscore_mean": mean,
        "zscore_std": std,
    }
    (cache_dir / f"{sid}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def run(cache_dir: Path | None = None, subjects: list[str] | None = None) -> None:
    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    subjects = subjects or list(ALL_SUBJECTS)
    metas = []
    for sid in subjects:
        print(f"preparing {sid}")
        metas.append(process_subject(sid, cache_dir))
    index = {
        "n": len(metas),
        "target_spacing_mm": TARGET_SPACING,
        "subjects": [m["subject_id"] for m in metas],
        "shapes": {m["subject_id"]: m["cached_shape"] for m in metas},
    }
    (cache_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"cached {len(metas)} subjects under {cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess FeTA volumes to .npy cache")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    run(cache_dir=args.cache_dir, subjects=args.subjects)


if __name__ == "__main__":
    main()
