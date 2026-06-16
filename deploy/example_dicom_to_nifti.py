"""DICOM series → canonical-RAS NIfTI converter for the worker.

The worker invokes `convert(dicom_dir, out_dir, study_id)` before any encoder
forward. The output `<out_dir>/<study_id>.nii.gz` is what the encoders'
preprocess paths read.

Backed by dcm2niix (preferred — battle-tested on Merlin data) with a nibabel
fallback. Always reorients to canonical RAS via nibabel after conversion so
the downstream encoder preprocessing sees a consistent axis convention.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib

log = logging.getLogger("ctvlm.dicom_to_nifti")

DCM2NIIX_BIN = os.environ.get("CTVLM_DCM2NIIX", "dcm2niix")


def convert(dicom_dir: str | Path, out_dir: str | Path, study_id: str) -> Path:
    """Convert a directory of .dcm files into a single canonical-RAS .nii.gz.

    Returns the output path. Raises FileNotFoundError if the input directory
    is empty or RuntimeError if dcm2niix produces no output.
    """
    dicom_dir = Path(dicom_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{study_id}.nii.gz"

    if not dicom_dir.exists() or not any(dicom_dir.iterdir()):
        raise FileNotFoundError(f"empty DICOM dir: {dicom_dir}")

    # ── 1. dcm2niix ─────────────────────────────────────────────────────── #
    if shutil.which(DCM2NIIX_BIN):
        _run_dcm2niix(dicom_dir, out_dir, study_id)
    else:
        # ── 2. Fallback: nibabel direct DICOM read ───────────────────────── #
        log.warning("dcm2niix not on PATH, falling back to nibabel — slower")
        _run_nibabel_fallback(dicom_dir, out_path)

    # ── 3. Reorient to canonical RAS so the encoder preprocess paths agree ─ #
    if not out_path.exists():
        # dcm2niix may have used a different filename when input didn't
        # follow the expected pattern. Pick the largest .nii.gz produced.
        candidates = sorted(out_dir.glob("*.nii.gz"), key=lambda p: -p.stat().st_size)
        if not candidates:
            raise RuntimeError(f"no NIfTI produced in {out_dir}")
        candidates[0].rename(out_path)

    img = nib.load(str(out_path))
    canon = nib.as_closest_canonical(img)
    nib.save(canon, str(out_path))
    log.info("wrote %s  shape=%s  spacing=%s",
             out_path, canon.shape,
             tuple(round(float(abs(canon.affine[i, i])), 2) for i in range(3)))
    return out_path


def _run_dcm2niix(dicom_dir: Path, out_dir: Path, study_id: str) -> None:
    """Run dcm2niix with sensible CT defaults. Single-series input expected."""
    cmd = [
        DCM2NIIX_BIN,
        "-f", study_id,        # output filename = study_id
        "-z", "y",             # gzip
        "-o", str(out_dir),
        "-b", "n",             # no JSON sidecar
        "-w", "1",             # overwrite if exists
        "-i", "y",             # ignore derivatives
        str(dicom_dir),
    ]
    log.info("dcm2niix: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        log.error("dcm2niix stdout: %s", proc.stdout[-2000:])
        log.error("dcm2niix stderr: %s", proc.stderr[-2000:])
        raise RuntimeError(
            f"dcm2niix exit {proc.returncode} on {dicom_dir}")


def _run_nibabel_fallback(dicom_dir: Path, out_path: Path) -> None:
    """Pure-Python DICOM → NIfTI. Slow but no system deps."""
    try:
        import pydicom                                           # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "no dcm2niix on PATH and pydicom not installed — "
            "install pydicom or dcm2niix to convert DICOM"
        ) from e
    import numpy as np
    from pydicom import dcmread

    files = sorted(dicom_dir.glob("*.dcm"))
    if not files:
        files = sorted(dicom_dir.rglob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no .dcm files in {dicom_dir}")

    slices = [dcmread(str(f)) for f in files]
    # Sort by ImagePositionPatient[2] (z) — robust to filename ordering
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    rows = int(slices[0].Rows)
    cols = int(slices[0].Columns)
    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)
    if vol.shape != (rows, cols, len(slices)):
        raise RuntimeError(f"unexpected vol shape {vol.shape}")

    # Rescale to HU using DICOM scale + intercept
    slope = float(slices[0].get("RescaleSlope", 1.0))
    inter = float(slices[0].get("RescaleIntercept", 0.0))
    vol = vol * slope + inter

    # Affine: row + column spacing from PixelSpacing, z from positions
    rs, cs = (float(x) for x in slices[0].PixelSpacing)
    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
    z_spacing = (z_positions[-1] - z_positions[0]) / max(len(z_positions) - 1, 1)
    affine = np.diag([cs, rs, z_spacing, 1.0]).astype(np.float32)
    origin = np.array(slices[0].ImagePositionPatient, dtype=np.float32)
    affine[:3, 3] = origin

    nib.save(nib.Nifti1Image(vol, affine), str(out_path))
