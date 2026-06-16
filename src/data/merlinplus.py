"""Load MerlinPlus organ masks for Merlin CT studies.

MerlinPlus (`AbdomenAtlas/MerlinPlus`, R-Super / Johns Hopkins) provides per-voxel
organ masks for the Merlin abdominal CT set. Task 11 validated it as the **primary
organ source** for Phase 1 (major-organ Dice 0.925 vs TotalSegmentator; see
`reports/merlinplus_vs_totalseg.md`). This module is the single entry point for
reading those masks.

Two facts every caller must respect — both handled here:
  1. **z-flip.** MerlinPlus masks are stored z-reversed relative to the Merlin CT
     (CT axcodes L,A,S; mask L,A,I — same physical space, opposite slice order).
     `load_mask` / `load_ct` reorient to canonical RAS via `nib.as_closest_canonical`
     so a mask and its CT line up voxel-for-voxel.
  2. **Model-generated, not ground truth.** These are R-Super predictions. Trust the
     major solid organs; spot-check the low-agreement classes (see LOW_AGREEMENT)
     before they feed any rule.

On-disk layout (set by the Task 11 extraction):
    <merlin_plus_dir>/extracted/<study_id>/segmentations/<class>.nii.gz

Public API:
    case_dir(study_id) -> Path
    has_case(study_id) -> bool
    list_cases() -> list[str]
    available_classes(study_id) -> list[str]
    load_mask(study_id, cls) -> np.ndarray            # canonical-RAS bool volume
    load_ct(study_id) -> nib.Nifti1Image | None       # canonical-RAS, matching grid
    qc_case(study_id) -> CaseQC
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nibabel as nib

from src.config import paths

# Merlin CT volumes (read-only). Study `AC4213...` -> <merlin_data>/<id>.nii.gz
_MERLIN_CT = paths.merlin_root / "merlinabdominalctdataset" / "merlin_data"


def _extracted_root() -> Path:
    return paths.merlin_plus_dir / "extracted"


# ---------------------------------------------------------------------------
# Class taxonomy
#
# 44 classes total. 36 appear in every case; 8 (CORE-but-occasionally-absent —
# vessels, ducts, the CBD stent) are missing when R-Super found nothing. Grouping
# is for Phase 3 node typing and the Phase 1 coverage check, not a hard schema.
# ---------------------------------------------------------------------------
SOLID_ORGANS = [
    "liver", "spleen", "kidney_left", "kidney_right", "pancreas",
    "gall_bladder", "adrenal_gland_left", "adrenal_gland_right",
    "bladder", "prostate", "stomach",
]
GI_TRACT = ["esophagus", "duodenum", "colon", "intestine", "rectum"]
VESSELS = [
    "aorta", "postcava", "portal_vein_and_splenic_vein", "hepatic_vessel",
    "celiac_trunk", "celiac_aa", "superior_mesenteric_artery",
    "renal_vein_left", "renal_vein_right", "veins",
]
DUCTS = ["common_bile_duct", "pancreatic_duct", "cbd_stent"]
LIVER_SEGMENTS = [f"liver_segment_{i}" for i in range(1, 9)]
PANCREAS_PARTS = ["pancreas_head", "pancreas_body", "pancreas_tail"]
SKELETAL = ["femur_left", "femur_right"]
THORACIC = ["lung_left", "lung_right"]

CLASS_GROUPS: dict[str, list[str]] = {
    "solid_organ": SOLID_ORGANS,
    "gi_tract": GI_TRACT,
    "vessel": VESSELS,
    "duct": DUCTS,
    "liver_segment": LIVER_SEGMENTS,
    "pancreas_part": PANCREAS_PARTS,
    "skeletal": SKELETAL,
    "thoracic": THORACIC,
}
# Authoritative flat list (44), and reverse class -> group lookup.
MERLINPLUS_CLASSES: list[str] = [c for g in CLASS_GROUPS.values() for c in g]
GROUP_OF: dict[str, str] = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}

# Classes Task 11 flagged as low MerlinPlus<->TotalSegmentator agreement
# (mean Dice < 0.8). Do not feed these to rules without a radiologist spot-check.
LOW_AGREEMENT = {
    "gall_bladder", "adrenal_gland_left", "adrenal_gland_right",
    "esophagus", "duodenum", "colon",
}

# The five organs whose mean Dice drove the Task 11 decision.
MAJOR_ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas"]


# ---------------------------------------------------------------------------
# Case access
# ---------------------------------------------------------------------------

def case_dir(study_id: str) -> Path:
    """Directory holding a case's per-class masks (may not exist)."""
    return _extracted_root() / study_id / "segmentations"


def has_case(study_id: str) -> bool:
    return case_dir(study_id).is_dir()


def list_cases() -> list[str]:
    """Study IDs with extracted MerlinPlus masks, sorted."""
    root = _extracted_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "segmentations").is_dir())


def available_classes(study_id: str) -> list[str]:
    """Classes present for a case, in canonical MERLINPLUS_CLASSES order.

    Returns only recognized classes; an unexpected filename is ignored (and
    surfaced by qc_case as `unexpected`).
    """
    d = case_dir(study_id)
    if not d.is_dir():
        return []
    present = {p.name[: -len(".nii.gz")] for p in d.glob("*.nii.gz")}
    return [c for c in MERLINPLUS_CLASSES if c in present]


def _canon(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient to closest canonical (RAS) orientation — undoes the z-flip."""
    return nib.as_closest_canonical(img)


def load_mask(study_id: str, cls: str) -> np.ndarray:
    """Load one class mask as a canonical-RAS boolean volume.

    Raises FileNotFoundError if the class is absent for this case (callers that
    treat absence as "empty" should check `available_classes` first).
    """
    if cls not in GROUP_OF:
        raise ValueError(f"unknown MerlinPlus class: {cls!r}")
    p = case_dir(study_id) / f"{cls}.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"{study_id}: class {cls!r} not present at {p}")
    arr = np.asanyarray(_canon(nib.load(str(p))).dataobj)
    return arr > 0


def ct_path(study_id: str) -> Path:
    """Path to the matching Merlin CT (may not exist). Cheap — no file open."""
    return _MERLIN_CT / f"{study_id}.nii.gz"


def load_ct(study_id: str) -> nib.Nifti1Image | None:
    """Load the matching Merlin CT, reoriented to canonical RAS (or None if absent).

    Reoriented the same way as the masks, so `load_ct(id)` and `load_mask(id, cls)`
    share a voxel grid.
    """
    p = ct_path(study_id)
    if not p.exists():
        return None
    return _canon(nib.load(str(p)))


# ---------------------------------------------------------------------------
# Per-case QC
# ---------------------------------------------------------------------------

@dataclass
class CaseQC:
    study_id: str
    present: list[str] = field(default_factory=list)      # recognized classes found
    missing: list[str] = field(default_factory=list)      # expected but absent
    unexpected: list[str] = field(default_factory=list)   # files we don't recognize
    empty: list[str] = field(default_factory=list)        # present but all-zero
    ct_exists: bool = False
    ct_shape: tuple | None = None
    shape_mismatch: list[str] = field(default_factory=list)  # mask grid != CT grid
    major_voxels: dict[str, int] = field(default_factory=dict)  # major-organ size

    @property
    def ok(self) -> bool:
        """A usable case: CT present, all major organs present, non-empty, on-grid."""
        if not self.ct_exists:
            return False
        majors_present = all(m in self.present for m in MAJOR_ORGANS)
        majors_clean = not any(m in self.empty or m in self.shape_mismatch
                               for m in MAJOR_ORGANS)
        return majors_present and majors_clean


def qc_case(study_id: str, check_shapes: bool = True) -> CaseQC:
    """Inspect one case: presence, empties, CT match, grid alignment, major sizes.

    `check_shapes=False` skips loading masks for shape/empty checks (presence-only,
    much faster for a first pass over many cases).
    """
    qc = CaseQC(study_id=study_id)
    d = case_dir(study_id)
    if not d.is_dir():
        qc.missing = list(MERLINPLUS_CLASSES)
        return qc

    on_disk = {p.name[: -len(".nii.gz")] for p in d.glob("*.nii.gz")}
    qc.present = [c for c in MERLINPLUS_CLASSES if c in on_disk]
    qc.missing = [c for c in MERLINPLUS_CLASSES if c not in on_disk]
    qc.unexpected = sorted(on_disk - set(MERLINPLUS_CLASSES))

    # Cheap existence check; only open the CT when we need its grid (check_shapes).
    qc.ct_exists = ct_path(study_id).exists()
    ref_shape = None
    if check_shapes and qc.ct_exists:
        ct = load_ct(study_id)
        ref_shape = ct.shape
        qc.ct_shape = tuple(int(x) for x in ref_shape)

    if check_shapes:
        for cls in qc.present:
            img = _canon(nib.load(str(d / f"{cls}.nii.gz")))
            if ref_shape is not None and img.shape != ref_shape:
                qc.shape_mismatch.append(cls)
            n = int((np.asanyarray(img.dataobj) > 0).sum())
            if n == 0:
                qc.empty.append(cls)
            if cls in MAJOR_ORGANS:
                qc.major_voxels[cls] = n
    return qc
