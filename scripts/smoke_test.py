"""Post-install smoke test for ctvlm.

Tests, in order:
  1. Importable: agent, embeddings, explain, data modules load.
  2. Canonical map: 220 unique canonicals across 17 categories.
  3. Probe checkpoint: loads with platt_A, platt_B, platt_A_nc, thresholds,
     val_aucs, source_dim=3200.
  4. Encoders (Merlin + Pillar-0) load from $HF_HOME without network access
     (skip if HF cache missing — prints SKIP, exits 0).
  5. Optional: if a sample (sid, NIfTI) is provided via --sid + --ct, run
     pipeline.generate_report end-to-end with skip_llm=True.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test.py
    python scripts/smoke_test.py --sid AC421363f --ct /path/to/AC421363f.nii.gz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _step(n: int, msg: str) -> None:
    print(f"\n[{n}] {msg}")


def step_imports() -> None:
    _step(1, "imports")
    from src.agent import pipeline, recipes, render, tools         # noqa: F401
    from src.agent.canonical import CANONICAL_NAMES, CATEGORY_TO_CANONICALS
    from src.embeddings import merlin, pillar0                     # noqa: F401
    from src.explain import cam                                    # noqa: F401
    print(f"  ok   ({len(CANONICAL_NAMES)} canonicals, "
          f"{len(CATEGORY_TO_CANONICALS)} categories)")


def step_canonical_map() -> None:
    _step(2, "canonical map")
    from src.agent.canonical import (CANONICAL_NAMES, CATEGORY_TO_CANONICALS,
                                      CANONICAL_TO_CATEGORY)
    # CANONICAL_NAMES is the 225-question RATE list (with 5 duplicates that
    # OR-collapse). CANONICAL_TO_CATEGORY is keyed by the 220 unique canonicals
    # the probe actually outputs.
    assert len(CANONICAL_TO_CATEGORY) == 220, \
        f"expected 220 unique canonicals, got {len(CANONICAL_TO_CATEGORY)}"
    assert len(CATEGORY_TO_CANONICALS) == 17, \
        f"expected 17 categories, got {len(CATEGORY_TO_CANONICALS)}"
    # Spot-check a Tier A anchor
    assert "hepatic_steatosis" in CANONICAL_TO_CATEGORY
    assert CANONICAL_TO_CATEGORY["hepatic_steatosis"] == "Liver"
    print(f"  ok   ({len(CANONICAL_TO_CATEGORY)} unique canonicals over "
          f"{len(CANONICAL_NAMES)} RATE questions)")


def step_probe_checkpoint() -> None:
    _step(3, "probe checkpoint")
    import torch
    from src.agent import pipeline
    ck = torch.load(pipeline.DEFAULT_PROBE, map_location="cpu", weights_only=False)
    for key in ("state_dict", "findings", "platt_A", "platt_B",
                "platt_A_nc", "platt_B_nc", "thresholds",
                "thresholds_nc", "val_aucs"):
        assert key in ck, f"checkpoint missing {key!r}"
    assert int(ck.get("source_dim", -1)) == 3200, "source_dim should be 3200 (concat probe)"
    print(f"  ok   ({len(ck['findings'])} findings, "
          f"n_nc_platt={ck.get('n_nc_platt', '?')}, "
          f"val_macro_matched={ck.get('val_macro_matched', '?'):.3f})")


def step_encoders() -> None:
    _step(4, "encoder offline load")
    hf = os.environ.get("HF_HOME", str(REPO_ROOT / "hf_cache"))
    if not Path(hf).exists() or not any(Path(hf).rglob("*.safetensors*")) and \
       not any(Path(hf).rglob("*.bin")):
        print(f"  SKIP — no HF cache at {hf}. Run scripts/download_weights.sh first.")
        return
    try:
        from src.embeddings import merlin, pillar0
        m = merlin.load_model()
        p = pillar0.load_model()
        print(f"  ok   merlin={type(m).__name__}  pillar0={type(p).__name__}")
    except Exception as e:                                          # noqa: BLE001
        print(f"  FAIL — {type(e).__name__}: {e}")
        raise


def step_end_to_end(sid: str, ct_path: str | None) -> None:
    _step(5, f"end-to-end report on {sid}")
    if ct_path:
        # Stage the CT under CTVLM_MERLIN_ROOT (default ./work/ct_volumes)
        from src.config import paths
        target = paths.merlin_root / f"{sid}.nii.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            import shutil
            shutil.copy2(ct_path, target)
            print(f"  staged CT → {target}")
    from src.agent import pipeline
    try:
        report = pipeline.generate_report(sid, skip_llm=True, max_findings=12)
        print(f"  encoder: {report.encoder}")
        print(f"  positives ({len(report.positives)}): "
              f"{', '.join(report.positives[:5])}{'...' if len(report.positives) > 5 else ''}")
        print(f"  latency: {report.latency_total_s:.2f}s")
        print("  summary:")
        for line in report.summary_text.splitlines()[:8]:
            print(f"    {line}")
    except FileNotFoundError as e:
        print(f"  SKIP — {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", help="run an end-to-end test on this study ID")
    ap.add_argument("--ct", help="path to a CT .nii.gz to stage under CTVLM_MERLIN_ROOT")
    args = ap.parse_args()

    step_imports()
    step_canonical_map()
    step_probe_checkpoint()
    step_encoders()
    if args.sid:
        step_end_to_end(args.sid, args.ct)

    print("\nctvlm smoke test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
