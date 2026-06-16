"""Per-organ 3D ROI crops + morphometrics from MerlinPlus masks (Phase 2 prep).

Phase 2 turns each segmented structure into a scanner-invariant embedding. The first
step (plan_1.md L145, L149) is the input to that encoder: for each organ, a fixed-context
3D crop of the CT around the organ, plus morphological features (volume, HU stats,
surface-to-volume). This module produces both.

It builds on `src.data.merlinplus`, which guarantees the CT and mask share a canonical-RAS
voxel grid (z-flip handled). When extracting many organs from one study, load the CT once
and pass it in (`ct_img=`) — the CT is ~100-150 MB, the masks are small.

Core (bbox, crop, window, morphometrics) is pure numpy and runs in any env. `resize`
needs scipy (lazy-imported) and is only for producing fixed-size encoder inputs.

Public API:
    bounding_box(mask, pad_vox) -> ((lo,hi) per axis) | None
    extract(study_id, cls, ct_img=None, pad_mm=10.0) -> ROICrop | None
    window_hu(arr, level, width, normalize=True) -> np.ndarray
    morphometrics(mask, ct, spacing) -> dict
    resize(arr, out_shape, order=1) -> np.ndarray            # requires scipy
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import nibabel as nib

from src.data import merlinplus as mp

# Abdomen soft-tissue display window (level, width) in HU. Standard radiology preset.
ABDOMEN_WINDOW = (40.0, 400.0)


@dataclass
class ROICrop:
    study_id: str
    cls: str
    ct: np.ndarray            # cropped CT in HU (float32)
    mask: np.ndarray          # cropped organ mask (bool), same shape as ct
    spacing: tuple[float, float, float]   # mm per axis (canonical RAS: x, y, z)
    bbox: tuple[tuple[int, int], ...]      # ((lo,hi) per axis) in full-volume indices

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mask.shape


def _pad_vox(pad_mm: float, spacing: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert an isotropic mm padding into per-axis voxel counts via spacing."""
    return tuple(int(round(pad_mm / s)) for s in spacing)  # type: ignore[return-value]


def bounding_box(mask: np.ndarray, pad_vox=(0, 0, 0)) -> tuple | None:
    """Per-axis (lo, hi) index range of the mask's nonzero extent, padded + clamped.

    `hi` is exclusive (slice-ready). Returns None for an empty mask.
    """
    if not mask.any():
        return None
    out = []
    for ax in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != ax)
        nz = np.where(mask.any(axis=other))[0]
        lo, hi = int(nz[0]), int(nz[-1]) + 1
        lo = max(0, lo - pad_vox[ax])
        hi = min(mask.shape[ax], hi + pad_vox[ax])
        out.append((lo, hi))
    return tuple(out)


def extract(study_id: str, cls: str, ct_img: nib.Nifti1Image | None = None,
            pad_mm: float = 10.0) -> ROICrop | None:
    """Crop the CT to the organ's bounding box plus `pad_mm` of context.

    Returns None if the CT or mask is absent, the grids disagree, or the mask is empty.
    Pass a preloaded `ct_img` to avoid reloading the CT per organ.
    """
    if ct_img is None:
        ct_img = mp.load_ct(study_id)
        if ct_img is None:
            return None
    try:
        mask = mp.load_mask(study_id, cls)
    except FileNotFoundError:
        return None
    ct = np.asanyarray(ct_img.dataobj).astype(np.float32)
    if mask.shape != ct.shape:
        return None
    spacing = tuple(float(z) for z in ct_img.header.get_zooms()[:3])
    bb = bounding_box(mask, _pad_vox(pad_mm, spacing))
    if bb is None:
        return None
    sl = tuple(slice(lo, hi) for lo, hi in bb)
    return ROICrop(study_id, cls, ct[sl].copy(), mask[sl].copy(), spacing, bb)


def window_hu(arr: np.ndarray, level: float = ABDOMEN_WINDOW[0],
              width: float = ABDOMEN_WINDOW[1], normalize: bool = True) -> np.ndarray:
    """Clip HU to a [level±width/2] window; optionally rescale to [0, 1] float32."""
    lo, hi = level - width / 2.0, level + width / 2.0
    out = np.clip(arr, lo, hi)
    if normalize:
        out = (out - lo) / (hi - lo)
    return out.astype(np.float32)


def _surface_voxels(mask: np.ndarray) -> int:
    """Count boundary voxels (in-mask voxels with >=1 non-mask 6-neighbour).

    Pure numpy; volume edges count as boundary (padded with False).
    """
    p = np.pad(mask, 1)  # constant False border
    interior = (
        mask
        & p[2:, 1:-1, 1:-1] & p[:-2, 1:-1, 1:-1]
        & p[1:-1, 2:, 1:-1] & p[1:-1, :-2, 1:-1]
        & p[1:-1, 1:-1, 2:] & p[1:-1, 1:-1, :-2]
    )
    return int(mask.sum() - interior.sum())


def morphometrics(mask: np.ndarray, ct: np.ndarray,
                  spacing: tuple[float, float, float]) -> dict:
    """Volume, intra-mask HU stats, extent, and surface-to-volume for one structure.

    `mask` and `ct` must share a grid (e.g. an ROICrop's .mask/.ct). Returns zeros for
    an empty mask.
    """
    vox_mm3 = float(np.prod(spacing))
    n = int(mask.sum())
    if n == 0:
        return {"n_voxels": 0, "volume_ml": 0.0, "mean_hu": float("nan"),
                "std_hu": float("nan"), "min_hu": float("nan"), "max_hu": float("nan"),
                "surface_voxels": 0, "surface_to_volume": float("nan"),
                "extent_mm": (0.0, 0.0, 0.0)}
    vals = ct[mask]
    bb = bounding_box(mask)
    extent = tuple(round((hi - lo) * spacing[i], 1) for i, (lo, hi) in enumerate(bb))
    surf = _surface_voxels(mask)
    return {
        "n_voxels": n,
        "volume_ml": round(n * vox_mm3 / 1000.0, 2),
        "mean_hu": round(float(vals.mean()), 1),
        "std_hu": round(float(vals.std()), 1),
        "min_hu": round(float(vals.min()), 1),
        "max_hu": round(float(vals.max()), 1),
        "surface_voxels": surf,
        "surface_to_volume": round(surf / n, 4),
        "extent_mm": extent,
    }


def resize(arr: np.ndarray, out_shape: tuple[int, ...], order: int = 1) -> np.ndarray:
    """Resample to a fixed shape for a fixed-input encoder (requires scipy).

    order=1 (trilinear) for CT intensity; order=0 (nearest) for label masks.
    """
    try:
        from scipy.ndimage import zoom
    except ImportError as e:  # noqa: F841
        raise ImportError("roi_crops.resize needs scipy (use the head_ct_triage env)")
    factors = tuple(o / s for o, s in zip(out_shape, arr.shape))
    return zoom(arr, factors, order=order)
