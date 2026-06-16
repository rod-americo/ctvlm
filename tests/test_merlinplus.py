"""Tests for src.data.merlinplus against the real extracted MerlinPlus masks.

Skips automatically if no cases have been extracted yet. The load-time test that
matters most is grid alignment: after canonical-RAS reorientation a mask and its CT
must share a voxel grid — that is what proves the z-flip is handled.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data import merlinplus as mp

CASES = mp.list_cases()
HAS_DATA = len(CASES) > 0
requires_data = pytest.mark.skipif(not HAS_DATA, reason="no extracted MerlinPlus cases")


# --- taxonomy (no data needed) -------------------------------------------------

def test_taxonomy_is_consistent():
    assert len(mp.MERLINPLUS_CLASSES) == 44
    assert len(set(mp.MERLINPLUS_CLASSES)) == 44, "duplicate class names"
    # every class maps to exactly one group, and groups partition the class list
    assert set(mp.GROUP_OF) == set(mp.MERLINPLUS_CLASSES)
    flat = [c for cs in mp.CLASS_GROUPS.values() for c in cs]
    assert sorted(flat) == sorted(mp.MERLINPLUS_CLASSES)
    assert set(mp.MAJOR_ORGANS) <= set(mp.MERLINPLUS_CLASSES)
    assert mp.LOW_AGREEMENT <= set(mp.MERLINPLUS_CLASSES)


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        mp.load_mask(CASES[0] if HAS_DATA else "x", "not_an_organ")


# --- data-backed ---------------------------------------------------------------

@requires_data
def test_available_classes_ordered_and_recognized():
    cls = mp.available_classes(CASES[0])
    assert cls, "case has no recognized classes"
    assert set(cls) <= set(mp.MERLINPLUS_CLASSES)
    # returned in canonical order
    assert cls == [c for c in mp.MERLINPLUS_CLASSES if c in set(cls)]


@requires_data
def test_load_mask_is_boolean_volume():
    arr = mp.load_mask(CASES[0], "liver")
    assert arr.dtype == bool
    assert arr.ndim == 3
    assert arr.sum() > 0, "liver mask unexpectedly empty for first case"


@requires_data
def test_mask_and_ct_share_grid():
    """The z-flip correctness invariant."""
    sid = CASES[0]
    ct = mp.load_ct(sid)
    assert ct is not None, "no CT for first case"
    liver = mp.load_mask(sid, "liver")
    assert liver.shape == ct.shape, (
        f"mask grid {liver.shape} != CT grid {ct.shape} — reorientation broken"
    )


@requires_data
def test_load_missing_class_raises():
    # find a class absent for some case, or skip if every case is complete
    for sid in CASES:
        present = set(mp.available_classes(sid))
        absent = [c for c in mp.MERLINPLUS_CLASSES if c not in present]
        if absent:
            with pytest.raises(FileNotFoundError):
                mp.load_mask(sid, absent[0])
            return
    pytest.skip("every case has all 44 classes; nothing absent to test")


@requires_data
def test_qc_case_first_case_usable():
    qc = mp.qc_case(CASES[0])
    assert qc.ct_exists
    assert qc.ct_shape is not None
    # major organs present and non-empty for a normal abdominal CT
    for organ in mp.MAJOR_ORGANS:
        assert organ in qc.present, f"{organ} missing"
        assert organ not in qc.empty, f"{organ} empty"
        assert organ not in qc.shape_mismatch, f"{organ} off-grid"
    assert qc.ok
