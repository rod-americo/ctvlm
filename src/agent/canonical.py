"""Frozen canonical-name lookups for the 225 RATE questions.

Loads `data/rate_canonical_map.csv` at import time and exposes:

  RATE_QUESTION_TO_CANONICAL: dict[full_question_text -> snake_case_canonical]
  CANONICAL_TO_CATEGORY:      dict[snake_case_canonical -> organ_category]
  CATEGORY_TO_CANONICALS:     dict[organ_category -> list[snake_case_canonical]]
  CANONICAL_NAMES:            list[str] in CSV order
  canonical(question_text)    helper with fuzzy fallback (first-30-char prefix)

The CSV was produced one-time by `scripts/38_make_canonical_map.py` (gpt-4o-mini,
~$0.05). Hand-curate the CSV — never re-run the LLM at inference time.

All downstream agentic code (`src/agent/{recipes,router,templates,...}`,
report-renderer, dashboard Trace tab) uses these names — never raw RATE
question text.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = REPO_ROOT / "data" / "rate_canonical_map.csv"


def _load():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"canonical map missing: {CSV_PATH}. Run scripts/38_make_canonical_map.py first.")
    df = pd.read_csv(CSV_PATH)
    required = {"question", "canonical", "organ_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"canonical map missing columns: {missing}")
    return df


_df = _load()

RATE_QUESTION_TO_CANONICAL: dict[str, str] = dict(zip(_df["question"], _df["canonical"]))
CANONICAL_TO_CATEGORY: dict[str, str] = dict(zip(_df["canonical"], _df["organ_category"]))
CANONICAL_NAMES: list[str] = list(_df["canonical"])

CATEGORY_TO_CANONICALS: dict[str, list[str]] = defaultdict(list)
for c, cat in zip(_df["canonical"], _df["organ_category"]):
    CATEGORY_TO_CANONICALS[cat].append(c)
CATEGORY_TO_CANONICALS = dict(CATEGORY_TO_CANONICALS)


@lru_cache(maxsize=512)
def canonical(question_text: str) -> str | None:
    """Question text -> canonical name. Returns None for unknown.

    Tries exact match first, then 30-char-prefix fallback (the RATE pipeline
    sometimes has trailing whitespace / minor punctuation variants).
    """
    q = str(question_text).strip()
    if q in RATE_QUESTION_TO_CANONICAL:
        return RATE_QUESTION_TO_CANONICAL[q]
    # prefix fallback
    pref = q.lower()[:30]
    for k, v in RATE_QUESTION_TO_CANONICAL.items():
        if k.lower()[:30] == pref:
            return v
    return None
