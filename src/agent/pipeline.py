"""Orchestrator — CT → probabilities → enrichment → structured text → prose.

`generate_report(sid)` is the single function the dashboard and CLI call. The
encoder forward + probe step is owned here (so swapping the probe checkpoint
later for the RATE-225 concat probe is a one-line change).

Phase 2B: defaults to `concat_rate_probe.pt` — 220-canonical RATE-225 head on
L2[Merlin || Pillar-0] (full 25k cohort). The probe checkpoint records its
`source_dim`; predict() branches on that to load just Merlin (2048) or the L2
concat (3200). The concat recipe MUST match scripts/41:load_features.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.agent import recipes, router, render as renderer
from src.agent.schema import FullReport
from src.config import paths


DEFAULT_PROBE = paths.checkpoints_dir / "concat_rate_probe.pt"
MERLIN_FEATURES = paths.work_root / "merlin_global"
PILLAR0_FEATURES = paths.work_root / "pillar0_emb"


_probe_cache: dict = {}


def _load_probe(probe_path: Path) -> dict:
    """Cache the parsed probe checkpoint per path — `torch.load` is ~100 ms and
    the eval script calls predict() thousands of times."""
    key = str(probe_path)
    if key not in _probe_cache:
        _probe_cache[key] = torch.load(probe_path, map_location="cpu",
                                       weights_only=False)
    return _probe_cache[key]


def _load_feature(sid: str, source_dim: int) -> tuple[np.ndarray, str]:
    """Build the inference-time feature vector matching the probe's source_dim.

    2048 → Merlin only. 3200 → L2[Merlin || Pillar-0] (same recipe as training).
    """
    m_path = MERLIN_FEATURES / f"{sid}.npy"
    if not m_path.exists():
        raise FileNotFoundError(f"no cached Merlin feature for {sid}")
    x_m = np.load(m_path).astype(np.float32)
    if source_dim == 2048:
        return x_m, "merlin"
    if source_dim == 3200:
        p_path = PILLAR0_FEATURES / f"{sid}.npy"
        if not p_path.exists():
            raise FileNotFoundError(f"no cached Pillar-0 feature for {sid}")
        x_p = np.load(p_path).astype(np.float32)
        x_m_l2 = x_m / max(float(np.linalg.norm(x_m)), 1e-6)
        x_p_l2 = x_p / max(float(np.linalg.norm(x_p)), 1e-6)
        return np.concatenate([x_m_l2, x_p_l2]), "merlin+pillar0_L2"
    raise ValueError(f"unsupported source_dim {source_dim}; only 2048 / 3200")


def predict(sid: str, probe_path: Path | None = None,
            contrast_phase: str = "ce",
            ) -> tuple[dict[str, float], list[str], str,
                       dict[str, float], dict[str, float]]:
    """Return (probabilities, findings, encoder_label, thresholds, val_aucs).

    `contrast_phase` ∈ {"ce", "nc"}. When "nc" and the checkpoint carries
    NC-specific Platt parameters, applies the NC-calibrated path and uses the
    NC-recomputed Youden-J thresholds. Falls back to default calibration for
    findings without an NC fit.
    """
    if probe_path is None:
        probe_path = DEFAULT_PROBE
    ck = _load_probe(probe_path)
    sd = ck["state_dict"]
    findings = ck["findings"]
    head_type = ck.get("head_type", "linear")
    if head_type == "linear":
        W = sd["weight"].numpy()
        b = sd.get("bias", torch.zeros(W.shape[0])).numpy()
        source_dim = int(ck.get("source_dim", W.shape[1]))
        x, encoder_label = _load_feature(sid, source_dim)
        raw_logits = x @ W.T + b
    elif head_type == "mlp":
        W1 = sd["0.weight"].numpy(); b1 = sd["0.bias"].numpy()
        W2 = sd["3.weight"].numpy(); b2 = sd["3.bias"].numpy()
        source_dim = int(ck.get("source_dim", W1.shape[1]))
        x, encoder_label = _load_feature(sid, source_dim)
        h = np.maximum(0.0, x @ W1.T + b1)
        raw_logits = h @ W2.T + b2
    else:
        raise ValueError(f"unknown head_type {head_type!r}")
    # Phase-aware Platt calibration. NC checkpoint fields (platt_A_nc / *_nc)
    # were added in Phase 2D — older checkpoints fall back to default Platt.
    if contrast_phase == "nc" and ck.get("platt_A_nc") is not None:
        platt_A = np.asarray(ck["platt_A_nc"])
        platt_B = np.asarray(ck["platt_B_nc"])
        thr_arr = ck.get("thresholds_nc", ck.get("thresholds"))
    else:
        platt_A = np.asarray(ck["platt_A"]) if ck.get("platt_A") is not None else None
        platt_B = np.asarray(ck["platt_B"]) if ck.get("platt_B") is not None else None
        thr_arr = ck.get("thresholds")
    if platt_A is not None and platt_B is not None:
        cal_logits = platt_A * raw_logits + platt_B
    else:
        cal_logits = raw_logits
    probs = 1.0 / (1.0 + np.exp(-cal_logits))
    auc_arr = ck.get("val_aucs")
    thresholds = ({f: float(t) for f, t in zip(findings, np.asarray(thr_arr))}
                  if thr_arr is not None else {})
    val_aucs = ({f: float(a) for f, a in zip(findings, np.asarray(auc_arr))}
                if auc_arr is not None else {})
    return ({f: float(p) for f, p in zip(findings, probs)},
            findings, encoder_label + (":nc" if contrast_phase == "nc" else ""),
            thresholds, val_aucs)


def generate_report(sid: str, *,
                    threshold: float = 0.5,
                    min_threshold: float = 0.20,
                    max_findings: int = 12,
                    probe_path: Path = DEFAULT_PROBE,
                    skip_llm: bool = False) -> FullReport:
    """The whole pipeline.

    Args:
      threshold: predictions ≥ this become "positive" → router fires
      max_findings: cap on positives rendered (top-N by probability) — protects
        the LLM's 160-token budget on multi-pathology cases
      skip_llm: if True, skip the model load + generate (useful for fast
        development iteration on tools/recipes/templates)
    """
    t0 = time.time()
    probs, findings, encoder, per_finding_thr, per_finding_auc = predict(sid, probe_path)

    # Effective per-finding floor:
    #   1) Youden-J calibrated threshold (or `threshold` fallback). Computed on
    #      PLATT-CALIBRATED probs, so this IS the operating point that
    #      maximises TPR-FPR for each finding individually.
    #   2) Universal min-threshold floor — caps how low Youden-J can take us.
    # AUROC-weighted floor is no longer applied: after Platt scaling, the
    # calibrated probability itself reflects per-finding reliability, so an
    # extra AUROC penalty would double-count.
    def _is_positive(f: str) -> bool:
        t = max(per_finding_thr.get(f, threshold), min_threshold)
        return probs[f] >= t

    candidates = sorted(
        ((f, probs[f]) for f in findings if _is_positive(f)),
        key=lambda kv: -kv[1],
    )
    # COREQUIRES + SUBSUMPTION + per-category cap (filter_positives now takes
    # probs to pick the top-K per category).
    candidate_names = [f for f, _ in candidates]
    kept = set(renderer.filter_positives(candidate_names, probs=probs))
    positives = [(f, p) for f, p in candidates if f in kept][:max_findings]

    structured = [router.route(sid, f, p) for f, p in positives]

    summary = renderer.assemble_summary(structured, probs=probs)

    if skip_llm:
        prose = "(LLM skipped; structured summary only)"
    else:
        try:
            prose = renderer.render_prose(summary)
        except Exception as e:
            prose = f"(LLM render failed: {type(e).__name__}: {e})"

    return FullReport(
        study_id=sid,
        probabilities=probs,
        positives=[f for f, _ in positives],
        structured=structured,
        summary_text=summary,
        prose=prose,
        threshold=threshold,
        encoder=encoder,
        probe_path=str(probe_path),
        latency_total_s=time.time() - t0,
    )
