# 07 — Calibration & operating points

The production probe carries **four calibration tables**, applied in this order at inference:

```
raw_logits = x @ W.T + b                              # linear probe
cal_logits = platt_A * raw_logits + platt_B           # Platt per-finding
probs      = sigmoid(cal_logits)                      # bounded [0, 1]
positive   = (prob >= max(youden_j_per_finding, 0.20)) # operating point
```

When `contrast_phase="nc"` is passed:

```
platt_A    = platt_A_nc       # NC-specific for 26 of 220 findings
platt_B    = platt_B_nc       # (others fall back to default)
youden_j   = thresholds_nc    # NC-recomputed Youden-J
```

## What lives in `concat_rate_probe.pt`

```python
ck = torch.load(probe_path, weights_only=False)
# Keys:
#   state_dict        — Linear or MLP weights
#   findings          — 220 canonical names (positional index)
#   platt_A           — (220,) fp32, default Platt slope
#   platt_B           — (220,) fp32, default Platt intercept
#   platt_A_nc        — (220,) fp32, NC-specific (== platt_A where no NC fit)
#   platt_B_nc        — (220,) fp32, NC-specific
#   thresholds        — (220,) fp32, default Youden-J ∈ [0.05, 0.95]
#   thresholds_nc     — (220,) fp32, NC-recomputed for 26 findings
#   val_aucs          — (220,) fp32, val-set AUROC per finding
#   val_mAUROC        — scalar, full-220 macro mAUROC
#   val_macro_matched — scalar, matched-11 macro mAUROC
#   epoch, source_dim, feature_source, feature, head_type, hidden_dim, dropout
#   n_train, n_val, n_nc_platt
```

`n_nc_platt = 26` is the audit field: 26 of 220 findings had ≥ 5 NC val positives and got their own Platt fit. The other 194 inherit the default calibration even under `contrast_phase="nc"`.

## Default operating point: micro F1 ≈ 0.529 (CE), 0.517 (NC)

| | micro P | micro R | F1 |
|---|---|---|---|
| CE (default cal) | 0.503 | 0.557 | 0.529 |
| NC (default cal) | 0.467 | 0.562 | 0.510 |
| **NC (phase-aware)** | **0.475** | **0.568** | **0.517** |

Numbers are from the 5,082-case val split. See [docs/10_PERFORMANCE.md](10_PERFORMANCE.md) for per-finding breakdown.

## How to adjust the operating point at deployment

You usually shouldn't — the per-finding Youden-J + floor was tuned to maximise F1. But if your use case wants more precision (e.g. you're worried about FP load on radiologists), bump the floor:

```python
pipeline.generate_report(sid, min_threshold=0.30)
```

The floor sweep on the full val:

| `min_threshold` | micro P | micro R | F1 |
|---|---|---|---|
| 0.15 | 0.397 | 0.601 | 0.478 |
| **0.20 (default)** | **0.503** | **0.557** | **0.529** |
| 0.30 | 0.503 | 0.459 | 0.480 |
| 0.40 | 0.568 | 0.388 | 0.461 |

The default (0.20) is near the F1 knee. Anything above 0.30 starts losing recall faster than removing FPs.

## How phase-aware calibration was fit

`scripts/41_merlin_rate_probe.py:_fit_platt`:

1. Train the probe on all 25,414 cases (RATE ∩ Merlin features ∩ Pillar-0 features).
2. Capture raw val-set logits.
3. **Default Platt**: per-finding `sklearn.LogisticRegression(C=1e4)` fit on all 5,082 val cases, for findings with ≥ 5 val positives. Save `platt_A`, `platt_B`.
4. **NC Platt**: load `data/noncontrast_sids.txt`, intersect with val SIDs, get NC val subset (~40 cases). For each finding with ≥ 5 NC val positives, fit a separate Platt — these end up at `platt_A_nc[j]`, `platt_B_nc[j]`. Findings without enough NC positives keep `platt_A_nc[j] = platt_A[j]`.
5. Re-compute Youden-J on **NC-Platt-calibrated** probs for the 26 NC-fit findings → `thresholds_nc`. Others inherit `thresholds`.

This is **post-hoc calibration**, not retraining — fast (~30 s on the 3090) and uses no GPU. To re-run after data refresh:

```bash
python scripts/41_merlin_rate_probe.py --feature concat
```

The script overwrites `concat_rate_probe.pt` in `$CTVLM_CHECKPOINTS_DIR`. **Production should load the new checkpoint without restarting** (the probe cache invalidates on path change; if path is the same, restart the worker).

## How to update the non-contrast SID list

`data/noncontrast_sids.txt` is a frozen snapshot: 190 study IDs the regex flagged as explicitly non-contrast in the Merlin reports.

To regenerate / extend (e.g. when you add new cases):

```python
# scripts/(your future script).py — pseudo-code
import re, pandas as pd
NC = r"non[\s\-]?contrast|without (iv |intravenous )?contrast|no (iv|intravenous) contrast"
CE = r"\b(iv|intravenous|with) contrast|post[\s\-]?contrast|nephrogram"
df  = pd.read_csv("data/rate_full/reports_25k.csv")
txt = df["Report Text"].fillna("").str.lower()
nc  = txt.str.contains(NC, regex=True) & ~txt.str.contains(CE, regex=True)
df.loc[nc, "Accession"].to_csv(
    "data/noncontrast_sids.txt", index=False, header=False)
```

Then retrain the probe; the NC Platt updates automatically.

## When phase awareness doesn't help

The NC-specific Platt only fits **26 high-prevalence findings**. For the other 194 findings, NC and CE share calibration. So phase awareness:

- Helps most on common findings where the NC training pool is large enough (renal_hypodensity, simple_renal_cyst, hepatic_steatosis, aortic_atherosclerosis, etc.)
- Doesn't help on rare findings (`acute_extravasation`, `wilms_tumor`, `splenic_pseudocyst`, ...)

If non-contrast becomes a primary use case, the right next step is to **collect more non-contrast training cases** — even a few hundred more would let many more findings get their own NC fit.
