"""Gate: every column in finding_labels_rate.csv resolves to a canonical name.

If this fails, Phase 4's router will silently drop findings instead of
templating them, so it's a hard pre-merge check.

    pytest tests/test_canonical.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.agent import canonical as C

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = REPO_ROOT / "reports" / "rate_full_25k" / "finding_labels_rate.csv"


def test_csv_exists():
    assert C.CSV_PATH.exists(), f"missing {C.CSV_PATH}"


def test_no_empty_canonicals():
    assert all(c and c.strip() for c in C.CANONICAL_NAMES), "empty canonical names exist"


def test_snake_case_format():
    pat = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")
    bad = [c for c in C.CANONICAL_NAMES if not pat.match(c)]
    assert not bad, f"non-snake_case canonical names: {bad[:5]}"


def test_every_label_column_resolves():
    """Every label CSV column (except study_id) maps to a non-None canonical."""
    if not LABELS_CSV.exists():
        pytest.skip(f"{LABELS_CSV} not present — Phase 2 hasn't landed")
    cols = pd.read_csv(LABELS_CSV, nrows=0).columns.tolist()
    finding_cols = [c for c in cols if c != "study_id"]
    unresolved = [q for q in finding_cols if C.canonical(q) is None]
    assert not unresolved, (
        f"{len(unresolved)} label columns don't map to a canonical name; first 5: "
        f"{unresolved[:5]}")


def test_every_category_populated():
    """All 17 organ categories appear in the canonical map."""
    expected = {
        "Pancreas", "Spleen", "Biliary Tree", "Liver", "Gallbladder",
        "Genitourinary", "Adrenal gland", "Gastrointestinal", "Device",
        "Peritoneum", "Great Vessel", "Retroperitoneum", "Multi Organs",
        "Male Pelvis", "Female pelvis", "Musculoskeletal", "Visible Thoracic",
    }
    actual = set(C.CATEGORY_TO_CANONICALS.keys())
    missing = expected - actual
    assert not missing, f"missing categories: {missing}"
