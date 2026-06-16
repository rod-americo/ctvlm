"""Tests for src.data.roi_crops.

Synthetic-geometry tests need no data. Data-backed tests extract a real liver crop and
sanity-check its volume against physiology; they skip if no MerlinPlus cases exist.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.data import roi_crops as rc
from src.data import merlinplus as mp

CASES = mp.list_cases()
HAS_DATA = len(CASES) > 0
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
requires_data = pytest.mark.skipif(not HAS_DATA, reason="no extracted MerlinPlus cases")


# --- synthetic geometry (no data) ----------------------------------------------

def test_bounding_box_and_padding():
    m = np.zeros((10, 10, 10), bool)
    m[3:6, 4:7, 5:8] = True
    assert rc.bounding_box(m) == ((3, 6), (4, 7), (5, 8))
    # pad clamps at volume bounds
    assert rc.bounding_box(m, (2, 2, 2)) == ((1, 8), (2, 9), (3, 10))
    assert rc.bounding_box(np.zeros((4, 4, 4), bool)) is None


def test_surface_voxels_solid_cube():
    # 5x5x5 solid cube: interior is the inner 3x3x3=27; surface = 125 - 27.
    cube = np.ones((5, 5, 5), bool)
    assert rc._surface_voxels(cube) == 125 - 27


def test_window_hu_normalizes_to_unit():
    arr = np.array([-1000, -160, 40, 240, 3000], dtype=np.float32)
    out = rc.window_hu(arr)  # abdomen window [-160, 240]
    assert out.min() == 0.0 and out.max() == 1.0
    assert abs(float(out[2]) - 0.5) < 1e-3  # level (40) -> mid
    raw = rc.window_hu(arr, normalize=False)
    assert raw.min() == -160.0 and raw.max() == 240.0


def test_morphometrics_synthetic_volume():
    m = np.zeros((20, 20, 20), bool)
    m[5:15, 5:15, 5:15] = True            # 1000 voxels
    ct = np.full(m.shape, 50.0, np.float32)
    feat = rc.morphometrics(m, ct, (2.0, 2.0, 2.0))   # 8 mm^3/voxel
    assert feat["n_voxels"] == 1000
    assert abs(feat["volume_ml"] - 1000 * 8 / 1000.0) < 1e-6   # 8.0 mL
    assert feat["mean_hu"] == 50.0 and feat["std_hu"] == 0.0
    assert feat["extent_mm"] == (20.0, 20.0, 20.0)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed in this env")
def test_resize_shape_and_mask_binary():
    arr = np.random.rand(8, 9, 10).astype(np.float32)
    assert rc.resize(arr, (16, 16, 16)).shape == (16, 16, 16)
    m = np.zeros((8, 8, 8), bool); m[2:6, 2:6, 2:6] = True
    out = rc.resize(m.astype(np.uint8), (16, 16, 16), order=0)
    assert set(np.unique(out)) <= {0, 1}   # nearest keeps labels binary


# --- data-backed ----------------------------------------------------------------

@requires_data
def test_extract_liver_crop_geometry():
    crop = rc.extract(CASES[0], "liver", pad_mm=10.0)
    assert crop is not None
    assert crop.ct.shape == crop.mask.shape == crop.shape
    assert crop.mask.dtype == bool and crop.mask.any()
    # crop is a sub-volume; each axis no larger than the full grid
    ct_img = mp.load_ct(CASES[0])
    assert all(c <= f for c, f in zip(crop.shape, ct_img.shape))


@requires_data
def test_liver_volume_is_physiological():
    crop = rc.extract(CASES[0], "liver")
    feat = rc.morphometrics(crop.mask, crop.ct, crop.spacing)
    # adult liver is roughly 1.0-2.5 L; allow a wide band to catch gross errors only
    assert 400.0 < feat["volume_ml"] < 4000.0, feat["volume_ml"]
    # liver parenchyma ~ soft tissue, tens of HU (not air, not bone)
    assert -50.0 < feat["mean_hu"] < 150.0, feat["mean_hu"]
