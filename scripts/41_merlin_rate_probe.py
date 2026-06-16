"""Train the RATE-225 probe (Phase 2A: Merlin-only; Phase 2B: L2 concat).

Maps the 230 RATE question columns → canonical snake_case names via
`data/rate_canonical_map.csv` (collapsing duplicates with logical OR), then
trains a single linear head on the full 25k cohort with the same
val_frac=0.2 / seed=42 split every prior probe in this project used.

Feature variants (`--feature`):
  merlin  — 2048-d Merlin global only (Phase 2A). Saves merlin_rate_probe.pt.
  concat  — 3200-d L2-norm [Merlin || Pillar-0] (Phase 2B winning variant
            from reports/concat_ablation.md). Saves concat_rate_probe.pt.

Output schema mirrors the existing `merlin_cam_probe.pt` so
`src/agent/pipeline.py:predict()` loads it unchanged. The pipeline branches
on `source_dim` (2048 vs 3200) to decide whether to compute the concat feature
at inference time.

Run:
  python scripts/41_merlin_rate_probe.py                       # Merlin-only
  python scripts/41_merlin_rate_probe.py --feature concat      # L2 concat
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import paths     # noqa: E402

MERLIN_DIR = Path("/mnt/e/ctvlm/merlin_global")
PILLAR0_DIR = Path("/mnt/e/ctvlm/pillar0_emb")
RATE_CSV = REPO_ROOT / "reports" / "rate_full_25k" / "finding_labels_rate.csv"
CANON_CSV = REPO_ROOT / "data" / "rate_canonical_map.csv"
NC_SIDS_FILE = REPO_ROOT / "data" / "noncontrast_sids.txt"
DEFAULT_OUT = {
    "merlin": paths.checkpoints_dir / "merlin_rate_probe.pt",
    "concat": paths.checkpoints_dir / "concat_rate_probe.pt",
}

# Matched-concept macro audit (same 11 anchors as merlin_baseline / concat_ablation).
MATCHED = ["ascites", "hepatic_steatosis", "splenomegaly", "cirrhosis",
           "hepatic_cyst", "simple_renal_cyst", "renal_stones", "pleural_effusion",
           "lymphadenopathy", "aortic_atherosclerosis", "acute_pancreatitis"]


def load_canonical_relabel(rate_cols: list[str]) -> tuple[list[str], list[list[int]]]:
    """Map RATE question columns → canonical names; collapse duplicates.

    Returns (canonical_names, src_cols_per_canonical) where each src_cols entry
    is a list of column indices in the input CSV that should be OR'd into the
    canonical label.
    """
    canon = pd.read_csv(CANON_CSV)
    q2c = dict(zip(canon["question"], canon["canonical"]))
    bucket: dict[str, list[int]] = defaultdict(list)
    unmapped = []
    for j, col in enumerate(rate_cols):
        c = q2c.get(col)
        if c is None:
            # 30-char prefix fallback (mirrors src/agent/canonical.py)
            prefix = col[:30].strip().lower()
            for q, cn in q2c.items():
                if q[:30].strip().lower() == prefix:
                    c = cn; break
        if c is None:
            unmapped.append(col)
            continue
        bucket[c].append(j)
    names = sorted(bucket.keys())
    print(f"  mapped {len(rate_cols)} RATE cols → {len(names)} unique canonicals; "
          f"unmapped: {len(unmapped)}")
    if unmapped:
        for col in unmapped[:5]:
            print(f"    [unmapped] {col!r}")
        if len(unmapped) > 5:
            print(f"    ... and {len(unmapped) - 5} more")
    return names, [bucket[n] for n in names]


def assemble_labels(Y_raw: np.ndarray, src_cols: list[list[int]]) -> np.ndarray:
    """OR-collapse per-canonical column groups."""
    out = np.zeros((Y_raw.shape[0], len(src_cols)), dtype=np.float32)
    for j, idxs in enumerate(src_cols):
        if len(idxs) == 1:
            out[:, j] = Y_raw[:, idxs[0]]
        else:
            out[:, j] = (Y_raw[:, idxs].sum(axis=1) > 0).astype(np.float32)
    return out


def split_indices(n: int, val_frac: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    return perm[n_val:], perm[:n_val]


def build_head(in_dim: int, out_dim: int, *,
               head_type: str = "linear",
               hidden_dim: int = 512,
               dropout: float = 0.3) -> nn.Module:
    """Linear or MLP-1 head over the concat feature.

    MLP: Linear(in→hidden) → ReLU → Dropout → Linear(hidden→out). State-dict
    keys land as `0.weight`, `0.bias`, `3.weight`, `3.bias` (the `nn.Dropout`
    has no weights), so the inference path can reconstruct it from in/hidden/out.
    """
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    raise ValueError(f"unknown head_type {head_type!r}")


def train(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, val: np.ndarray,
          *, epochs: int = 80, lr: float = 1e-3, wd: float = 1e-4,
          batch: int = 64, dev: str = "cuda",
          head_type: str = "linear",
          hidden_dim: int = 512,
          dropout: float = 0.3):
    pos = Y[tr].sum(0)
    neg = len(tr) - pos
    pw = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 0.5, 20.0)).float().to(dev)
    Xt = torch.from_numpy(X[tr]).to(dev); Yt = torch.from_numpy(Y[tr]).to(dev)
    Xv = torch.from_numpy(X[val]).to(dev); Yv = Y[val]
    head = build_head(X.shape[1], Y.shape[1],
                       head_type=head_type, hidden_dim=hidden_dim,
                       dropout=dropout).to(dev)
    print(f"  head: {head_type} ("
          f"{sum(p.numel() for p in head.parameters())/1e6:.2f}M params)")
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_m, best_sd, best_ep, best_P, best_logits = -1.0, None, 0, None, None
    for ep in range(1, epochs + 1):
        head.train()
        idx = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), batch):
            ix = idx[i:i + batch]
            opt.zero_grad()
            loss = lossf(head(Xt[ix]), Yt[ix])
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == epochs:
            head.eval()
            with torch.no_grad():
                raw = head(Xv).cpu().numpy()
            P = 1.0 / (1.0 + np.exp(-raw))
            aucs = [roc_auc_score(Yv[:, j], P[:, j]) for j in range(Y.shape[1])
                    if 0 < Yv[:, j].sum() < len(Yv)]
            m = float(np.mean(aucs))
            if m > best_m:
                best_m = m
                best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                best_ep = ep
                best_P = P
                best_logits = raw
            print(f"  ep {ep:3d}  macro_mAUROC={m:.4f}  best={best_m:.4f} @ ep {best_ep}")
    return best_sd, best_m, best_ep, best_P, Yv, best_logits


def load_features(sids: list[str], feature: str) -> tuple[np.ndarray, str]:
    """Load per-encoder features and (optionally) build the L2-norm concat.

    The concat recipe MUST match `src/agent/pipeline.py:predict()` at inference
    time: per-encoder L2-norm, then concatenate Merlin first, Pillar-0 second.
    """
    print(f"  loading Merlin features ({len(sids)} cases) ...")
    X_m = np.stack([np.load(MERLIN_DIR / f"{s}.npy").astype(np.float32) for s in sids])
    if feature == "merlin":
        return X_m, "merlin_global_2048 (RATE-225)"
    print(f"  loading Pillar-0 features ({len(sids)} cases) ...")
    X_p = np.stack([np.load(PILLAR0_DIR / f"{s}.npy").astype(np.float32) for s in sids])
    print(f"  L2 mean: Merlin={np.linalg.norm(X_m, axis=1).mean():.2f}  "
          f"Pillar-0={np.linalg.norm(X_p, axis=1).mean():.2f}")
    X_m_l2 = X_m / np.clip(np.linalg.norm(X_m, axis=1, keepdims=True), 1e-6, None)
    X_p_l2 = X_p / np.clip(np.linalg.norm(X_p, axis=1, keepdims=True), 1e-6, None)
    X = np.concatenate([X_m_l2, X_p_l2], axis=1)
    return X, "L2[merlin_global_2048 || pillar0_emb_1152] (RATE-225)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=("merlin", "concat"), default="merlin",
                    help="merlin (2048-d, Phase 2A) or concat (3200-d L2, Phase 2B)")
    ap.add_argument("--head", choices=("linear", "mlp"), default="linear",
                    help="linear (2048|3200 → 220) or mlp (... → hidden → 220)")
    ap.add_argument("--hidden-dim", type=int, default=512,
                    help="MLP hidden layer width (ignored for linear head)")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="MLP hidden-layer dropout (ignored for linear head)")
    ap.add_argument("--out", default=None, help="override default output path")
    args = ap.parse_args()
    out_ckpt = Path(args.out) if args.out else DEFAULT_OUT[args.feature]
    if args.head == "mlp" and args.out is None:
        out_ckpt = out_ckpt.with_name(
            out_ckpt.stem + f"_mlp{args.hidden_dim}" + out_ckpt.suffix)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  feature: {args.feature}  out: {out_ckpt}")

    print("loading RATE label matrix ...")
    rate = pd.read_csv(RATE_CSV).set_index("study_id")
    rate_cols = list(rate.columns)
    print(f"  {len(rate)} cases × {len(rate_cols)} question columns")

    print("building canonical relabeling ...")
    canon_names, src_cols = load_canonical_relabel(rate_cols)

    print("filtering to study_ids with required features on disk ...")
    feat_set = {p.stem for p in MERLIN_DIR.glob("*.npy")}
    if args.feature == "concat":
        feat_set &= {p.stem for p in PILLAR0_DIR.glob("*.npy")}
    sids = sorted(rate.index.intersection(feat_set))
    print(f"  {len(sids)} cases (RATE ∩ features)")

    X, feature_source = load_features(sids, args.feature)
    Y_raw = rate.loc[sids, rate_cols].values.astype(np.float32)
    Y = assemble_labels(Y_raw, src_cols)
    print(f"  X: {X.shape}  Y: {Y.shape}  positives per finding (median): "
          f"{int(np.median(Y.sum(0)))}")

    tr, val = split_indices(len(sids))
    print(f"  train {len(tr)} / val {len(val)}")

    print("training ...")
    sd, best_m, best_ep, P_raw, Yv, raw_logits = train(
        X, Y, tr, val, dev=dev,
        head_type=args.head, hidden_dim=args.hidden_dim, dropout=args.dropout)

    # Per-finding Platt scaling: fit `p_cal = sigmoid(A*logit + B)` on val.
    # Calibrates each finding's sigmoid so a probability of 0.7 actually means
    # ≈70% positive rate at that operating point. Catches the systematic
    # over-confidence pattern that AUROC-weighting can't see (AUROC is invariant
    # under monotonic transforms; calibration is the inverse transform).
    print("\nfitting Platt scaling per finding ...")
    from sklearn.linear_model import LogisticRegression

    def _fit_platt(raw_logits_arr: np.ndarray, Y_arr: np.ndarray,
                   min_pos: int = 5) -> tuple[np.ndarray, np.ndarray, int]:
        """Fit per-finding Platt scaling (sigmoid(A*x + B)) using sklearn LR.
        Returns (A, B, n_calibrated). Findings with n_pos<min_pos or degenerate
        (A<0.1) keep A=1, B=0 (no-op calibration)."""
        A = np.ones(Y_arr.shape[1], dtype=np.float32)
        B = np.zeros(Y_arr.shape[1], dtype=np.float32)
        n_ok = 0
        for j in range(Y_arr.shape[1]):
            n_pos = int(Y_arr[:, j].sum())
            if n_pos < min_pos or n_pos == len(Y_arr):
                continue
            lr_cal = LogisticRegression(C=1e4, fit_intercept=True,
                                         solver="lbfgs", max_iter=500)
            try:
                lr_cal.fit(raw_logits_arr[:, j].reshape(-1, 1),
                           Y_arr[:, j].astype(int))
            except Exception:                       # noqa: BLE001
                continue
            a = float(lr_cal.coef_[0, 0])
            if a < 0.1:
                continue
            A[j] = a
            B[j] = float(lr_cal.intercept_[0])
            n_ok += 1
        return A, B, n_ok

    platt_A, platt_B, n_platt = _fit_platt(raw_logits, Yv)
    print(f"  default Platt: fit on {n_platt}/{Y.shape[1]} findings; "
          f"A median={np.median(platt_A):.3f}, B median={np.median(platt_B):.3f}")

    # Phase-stratified Platt: fit a separate calibration on the NON-CONTRAST
    # val subset for findings with enough NC positives. Falls back to the
    # default Platt elsewhere. NC training corpus is ~0.75% of cases, so we
    # only fit for the prevalent findings; rare ones inherit the global fit.
    platt_A_nc = platt_A.copy()
    platt_B_nc = platt_B.copy()
    n_platt_nc = 0
    if NC_SIDS_FILE.exists():
        nc_sids_all = set(NC_SIDS_FILE.read_text().strip().split("\n"))
        val_sids = [sids[i] for i in val]
        nc_val_mask = np.array([s in nc_sids_all for s in val_sids])
        n_nc_val = int(nc_val_mask.sum())
        print(f"  found {n_nc_val} non-contrast val cases (loading {NC_SIDS_FILE.name})")
        if n_nc_val >= 20:
            raw_nc = raw_logits[nc_val_mask]
            Y_nc = Yv[nc_val_mask]
            A_nc, B_nc, n_platt_nc = _fit_platt(raw_nc, Y_nc, min_pos=5)
            # Only overwrite where NC fit actually ran (A!=1 OR B!=0)
            fit_mask = (A_nc != 1.0) | (B_nc != 0.0)
            platt_A_nc[fit_mask] = A_nc[fit_mask]
            platt_B_nc[fit_mask] = B_nc[fit_mask]
            print(f"  NC-specific Platt: fit on {n_platt_nc}/{Y.shape[1]} findings; "
                  f"others fall back to default Platt")
        else:
            print(f"  too few NC val cases ({n_nc_val} < 20); skipping NC Platt")
    else:
        print(f"  no NC sid list at {NC_SIDS_FILE}; default Platt only")

    # Apply default calibration to get the production probabilities
    cal_logits = platt_A * raw_logits + platt_B
    P = 1.0 / (1.0 + np.exp(-cal_logits))

    # Per-finding Youden-J threshold on the val set: argmax_t (TPR(t) - FPR(t)).
    # Cheap (one np.argsort per finding) and gives every finding its own operating
    # point so downstream threshold=0.5 stops missing low-base-rate positives.
    print("\ncalibrating per-finding thresholds (Youden's J on val) + AUROCs ...")
    from sklearn.metrics import roc_curve

    def _fit_youden(P_arr: np.ndarray, Y_arr: np.ndarray,
                    min_pos: int = 5) -> np.ndarray:
        thr_out = np.full(Y_arr.shape[1], 0.5, dtype=np.float32)
        for j in range(Y_arr.shape[1]):
            n_pos = int(Y_arr[:, j].sum())
            if n_pos < min_pos:
                continue
            fpr, tpr, thr = roc_curve(Y_arr[:, j], P_arr[:, j])
            if len(thr) <= 1:
                continue
            j_stat = tpr - fpr
            best = int(np.argmax(j_stat))
            thr_out[j] = float(np.clip(thr[best], 0.05, 0.95))
        return thr_out

    thresholds = _fit_youden(P, Yv)
    val_aucs = np.full(Y.shape[1], 0.5, dtype=np.float32)
    for j in range(Y.shape[1]):
        if 0 < Yv[:, j].sum() < len(Yv):
            val_aucs[j] = float(roc_auc_score(Yv[:, j], P[:, j]))
    n_calibrated = int((thresholds != 0.5).sum())
    print(f"  default thresholds: calibrated {n_calibrated}/{Y.shape[1]} findings; "
          f"median = {np.median(thresholds):.3f}; "
          f"AUROC quartiles 25/50/75 = "
          f"{np.percentile(val_aucs, 25):.3f}/"
          f"{np.percentile(val_aucs, 50):.3f}/"
          f"{np.percentile(val_aucs, 75):.3f}")

    # NC-specific thresholds: recompute Youden-J on NC-Platt-calibrated probs
    # for findings that have an NC-specific Platt. Findings without NC Platt
    # keep the default thresholds.
    thresholds_nc = thresholds.copy()
    if NC_SIDS_FILE.exists() and n_platt_nc > 0:
        nc_sids_all = set(NC_SIDS_FILE.read_text().strip().split("\n"))
        val_sids = [sids[i] for i in val]
        nc_val_mask = np.array([s in nc_sids_all for s in val_sids])
        if nc_val_mask.sum() >= 20:
            cal_logits_nc = platt_A_nc * raw_logits[nc_val_mask] + platt_B_nc
            P_nc = 1.0 / (1.0 + np.exp(-cal_logits_nc))
            thr_nc_fit = _fit_youden(P_nc, Yv[nc_val_mask])
            # Only overwrite for findings that actually got an NC fit
            nc_fit_mask = (platt_A_nc != platt_A) | (platt_B_nc != platt_B)
            thresholds_nc[nc_fit_mask] = thr_nc_fit[nc_fit_mask]
            print(f"  NC thresholds: recomputed on {int(nc_fit_mask.sum())} NC-Platt findings; "
                  f"median (NC-fit only) = "
                  f"{np.median(thr_nc_fit[nc_fit_mask]) if nc_fit_mask.any() else float('nan'):.3f}")

    # Matched-concept audit
    matched_aucs = []
    for c in MATCHED:
        if c not in canon_names:
            print(f"  [WARN] matched anchor missing from canonical names: {c}")
            continue
        j = canon_names.index(c)
        if Yv[:, j].sum() == 0:
            continue
        a = roc_auc_score(Yv[:, j], P[:, j])
        matched_aucs.append((c, int(Yv[:, j].sum()), a))
    if matched_aucs:
        macro_matched = float(np.mean([a for *_, a in matched_aucs]))
        print(f"\n  matched-concept macro (11 anchors): {macro_matched:.4f}")
        for c, n, a in matched_aucs:
            print(f"    {c:<25}  n+={n:4d}  AUROC={a:.3f}")
    else:
        macro_matched = float("nan")

    ckpt = {
        "state_dict": sd,
        "findings": canon_names,
        "platt_A": platt_A,                 # default Platt slope (fp32)
        "platt_B": platt_B,                 # default Platt intercept (fp32)
        "platt_A_nc": platt_A_nc,           # NC-specific Platt slope (fp32)
        "platt_B_nc": platt_B_nc,           # NC-specific Platt intercept (fp32)
        "thresholds": thresholds,           # Youden-J on default-calibrated probs
        "thresholds_nc": thresholds_nc,     # Youden-J on NC-calibrated probs
        "n_nc_platt": int(n_platt_nc),
        "val_aucs": val_aucs,               # per-finding AUROC (fp32)
        "val_mAUROC": best_m,
        "val_macro_matched": macro_matched,
        "epoch": best_ep,
        "source_dim": X.shape[1],
        "feature_source": f"{feature_source} → {len(canon_names)} canonicals",
        "feature": args.feature,
        "head_type": args.head,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
    }
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_ckpt)
    print(f"\nwrote {out_ckpt}  (best macro={best_m:.4f} @ ep {best_ep})")


if __name__ == "__main__":
    main()
