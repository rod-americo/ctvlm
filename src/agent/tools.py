"""Pure-function tool registry the router calls per-finding.

Every tool: takes a `sid` and a small kwargs dict; returns a JSON-serialisable
dict (or a path string + summary stats for numpy outputs). Wrapped by
`src/agent/cache.py` so identical calls within a study return instantly from
disk.

Tools never call other tools — composition lives in the recipes. The few that
need a CT array or organ mask read those directly (small enough to re-load).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.data import merlinplus as mp
from src.data import roi_crops
from src.config import paths

# Heatmaps are produced by ct_files_server's lazy path or scripts/34; cached
# at $CTVLM_WORK_ROOT/heatmaps/<encoder>/<sid>/<finding>.nii.gz.
def heat_root() -> Path:
    raw = os.environ.get("CTVLM_WORK_ROOT")
    root = Path(raw).expanduser() if raw else paths.work_root
    return root / "heatmaps"


# --------------------------------------------------------------------------- #
# CT / organ-mask basics
# --------------------------------------------------------------------------- #

def get_ct_meta(sid: str) -> dict:
    """Voxel spacing, shape, axcodes — small metadata, not the array."""
    ct = mp.load_ct(sid)
    if ct is None:
        return {"present": False}
    ct = nib.as_closest_canonical(ct)
    return {
        "present": True,
        "shape": list(ct.shape),
        "spacing_mm": [float(abs(ct.affine[i, i])) for i in range(3)],
        "axcodes": list(nib.orientations.aff2axcodes(ct.affine)),
        "affine": ct.affine.tolist(),
    }


def organ_present(sid: str, organ_class: str) -> dict:
    """Cheap existence check before running morphometrics."""
    try:
        m = mp.load_mask(sid, organ_class)
        n = int(m.sum())
        return {"present": n > 0, "n_voxels": n}
    except FileNotFoundError:
        return {"present": False, "n_voxels": 0}


def organ_morphometrics(sid: str, organ_class: str) -> dict:
    """volume_ml, mean_hu, std_hu, extent_mm — everything roi_crops returns."""
    try:
        mask = mp.load_mask(sid, organ_class)
    except FileNotFoundError:
        return {"present": False}
    ct = nib.as_closest_canonical(mp.load_ct(sid))
    spacing = tuple(float(abs(ct.affine[i, i])) for i in range(3))
    m = roi_crops.morphometrics(mask, np.asanyarray(ct.dataobj), spacing)
    # Convert any non-JSON-friendly values
    m = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else
             (list(v) if isinstance(v, tuple) else v))
         for k, v in m.items()}
    m["present"] = True
    return m


def liver_to_spleen_hu_ratio(sid: str) -> dict:
    """Standard radiologic diagnostic for hepatic steatosis."""
    liver = organ_morphometrics(sid, "liver")
    spleen = organ_morphometrics(sid, "spleen")
    if not (liver.get("present") and spleen.get("present")):
        return {"valid": False}
    l_hu = liver.get("mean_hu")
    s_hu = spleen.get("mean_hu")
    if l_hu is None or s_hu is None:
        return {"valid": False}
    return {
        "valid": True,
        "liver_mean_hu": float(l_hu),
        "spleen_mean_hu": float(s_hu),
        "liver_minus_spleen_hu": float(l_hu - s_hu),
        "steatosis_likely": bool((l_hu - s_hu) < -10),
    }


# --------------------------------------------------------------------------- #
# CAM heatmap consumers (the heatmap NIfTIs already exist; this is read-only)
# --------------------------------------------------------------------------- #

def heatmap_path(sid: str, encoder: str, finding: str) -> Path:
    return heat_root() / encoder / sid / f"{finding}.nii.gz"


def cam_peak(sid: str, encoder: str, finding: str) -> dict:
    """Peak voxel + world-space coord + axial slice from cached heatmap."""
    p = heatmap_path(sid, encoder, finding)
    if not p.exists():
        from src.explain import cam as _cam
        generated = _cam.ensure_concat_heatmap(sid, encoder, finding,
                                                heat_root=heat_root(), verbose=True)
        if generated is None or not generated.exists():
            return {"valid": False, "reason": "heatmap not cached"}
        p = generated
    img = nib.load(str(p))
    arr = np.asanyarray(img.dataobj)
    if not (arr > 0).any():
        return {"valid": False, "reason": "all-zero heatmap"}
    flat_idx = int(arr.argmax())
    vox = np.unravel_index(flat_idx, arr.shape)
    world = img.affine @ np.array([*vox, 1.0])
    return {
        "valid": True,
        "voxel_idx": [int(v) for v in vox],
        "world_mm": [float(world[0]), float(world[1]), float(world[2])],
        "axial_slice": int(vox[2]) + 1,           # 1-indexed for clinical convention
        "peak_value": float(arr[vox]),
    }


def cam_connected_components(sid: str, encoder: str, finding: str,
                             threshold: float = 0.5) -> dict:
    """Connected-component extent of the activated region (Tier A focal lesions)."""
    from scipy.ndimage import label
    p = heatmap_path(sid, encoder, finding)
    if not p.exists():
        from src.explain import cam as _cam
        generated = _cam.ensure_concat_heatmap(sid, encoder, finding,
                                                heat_root=heat_root(), verbose=True)
        if generated is None or not generated.exists():
            return {"valid": False, "reason": "heatmap not cached"}
        p = generated
    img = nib.load(str(p))
    arr = np.asanyarray(img.dataobj)
    spacing = [float(abs(img.affine[i, i])) for i in range(3)]
    binary = arr > threshold
    if not binary.any():
        return {"valid": False, "reason": "no voxels above threshold"}
    labels, n = label(binary)
    out = []
    for cc_id in range(1, min(n + 1, 6)):           # keep top 5 by size
        mask = labels == cc_id
        nvox = int(mask.sum())
        if nvox < 5:
            continue
        idxs = np.argwhere(mask)
        centroid_vox = idxs.mean(0).tolist()
        ext_vox = (idxs.max(0) - idxs.min(0) + 1).tolist()
        ext_mm = [float(ext_vox[i] * spacing[i]) for i in range(3)]
        out.append({
            "n_vox": nvox,
            "centroid_vox": [float(v) for v in centroid_vox],
            "extent_mm": ext_mm,
        })
    out.sort(key=lambda r: -r["n_vox"])
    return {"valid": True, "components": out, "n_total": len(out)}


def axial_slice_of(voxel_idx: list[int]) -> int:
    """Translate a (R, A, S) voxel index to a 1-indexed axial slice number."""
    return int(voxel_idx[2]) + 1


# --------------------------------------------------------------------------- #
# Sub-organ localisation (the MerlinPlus segment masks)
# --------------------------------------------------------------------------- #

def liver_segment_at(sid: str, voxel_idx: list[int] | None) -> dict:
    """Which Couinaud segment (1-8) contains this voxel? Vote across the 8 masks."""
    if voxel_idx is None:
        return {"valid": False, "reason": "no voxel_idx (upstream cam_peak failed)"}
    vox = tuple(int(v) for v in voxel_idx)
    counts = {}
    for i in range(1, 9):
        try:
            m = mp.load_mask(sid, f"liver_segment_{i}")
        except FileNotFoundError:
            continue
        try:
            counts[i] = int(bool(m[vox]))
        except IndexError:
            counts[i] = 0
    hits = [i for i, c in counts.items() if c]
    if not hits:
        return {"valid": False, "reason": "voxel outside all liver segments"}
    ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}
    return {"valid": True, "segment": int(hits[0]),
            "roman": ROMAN[hits[0]], "candidates": hits}


def pancreas_region_at(sid: str, voxel_idx: list[int] | None) -> dict:
    if voxel_idx is None:
        return {"valid": False, "reason": "no voxel_idx"}
    vox = tuple(int(v) for v in voxel_idx)
    for region in ("pancreas_head", "pancreas_body", "pancreas_tail"):
        try:
            m = mp.load_mask(sid, region)
        except FileNotFoundError:
            continue
        try:
            if m[vox]:
                return {"valid": True, "region": region.split("_")[-1]}
        except IndexError:
            pass
    return {"valid": False, "reason": "voxel outside all pancreas regions"}


def kidney_side_at(sid: str, voxel_idx: list[int] | None) -> dict:
    if voxel_idx is None:
        return {"valid": False, "reason": "no voxel_idx"}
    vox = tuple(int(v) for v in voxel_idx)
    for side in ("kidney_left", "kidney_right"):
        try:
            m = mp.load_mask(sid, side)
        except FileNotFoundError:
            continue
        try:
            if m[vox]:
                return {"valid": True, "side": side.split("_")[1]}
        except IndexError:
            pass
    return {"valid": False, "reason": "voxel outside both kidney masks"}


def lesion_in_organ(sid: str, voxel_idx: list[int] | None, organ_class: str) -> dict:
    """Does the given voxel sit inside the named organ's mask?"""
    if voxel_idx is None:
        return {"valid": False, "reason": "no voxel_idx"}
    vox = tuple(int(v) for v in voxel_idx)
    try:
        m = mp.load_mask(sid, organ_class)
    except FileNotFoundError:
        return {"valid": False, "reason": "no mask"}
    try:
        return {"valid": True, "in_organ": bool(m[vox])}
    except IndexError:
        return {"valid": False, "reason": "voxel out of bounds"}


def sample_hu_at(sid: str, voxel_idx: list[int] | None, radius_vox: int = 2) -> dict:
    """Window the original CT around voxel_idx and return basic HU stats."""
    if voxel_idx is None:
        return {"valid": False, "reason": "no voxel_idx"}
    ct = nib.as_closest_canonical(mp.load_ct(sid))
    arr = np.asanyarray(ct.dataobj)
    vi, vj, vk = (int(v) for v in voxel_idx)
    lo = lambda v, r: max(0, v - r)
    hi = lambda v, r, n: min(n, v + r + 1)
    patch = arr[lo(vi, radius_vox):hi(vi, radius_vox, arr.shape[0]),
                lo(vj, radius_vox):hi(vj, radius_vox, arr.shape[1]),
                lo(vk, radius_vox):hi(vk, radius_vox, arr.shape[2])]
    if patch.size == 0:
        return {"valid": False, "reason": "empty patch"}
    return {
        "valid": True,
        "mean_hu": float(patch.mean()),
        "std_hu": float(patch.std()),
        "min_hu": float(patch.min()),
        "max_hu": float(patch.max()),
        "n_voxels": int(patch.size),
    }


# --------------------------------------------------------------------------- #
# Tool registry — name → callable (for the router)
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, Any] = {
    "get_ct_meta":             get_ct_meta,
    "organ_present":           organ_present,
    "organ_morphometrics":     organ_morphometrics,
    "liver_to_spleen_hu_ratio": liver_to_spleen_hu_ratio,
    "cam_peak":                cam_peak,
    "cam_connected_components": cam_connected_components,
    "axial_slice_of":          axial_slice_of,
    "liver_segment_at":        liver_segment_at,
    "pancreas_region_at":      pancreas_region_at,
    "kidney_side_at":          kidney_side_at,
    "lesion_in_organ":         lesion_in_organ,
    "sample_hu_at":            sample_hu_at,
}
