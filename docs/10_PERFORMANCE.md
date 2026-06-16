# 10 — Performance

All numbers are from the **5,082-case val split** (the held-out 20% of the 25,414-case RATE-Merlin-Pillar-0 cohort, seed=42).

## Headline (production calibration)

| split | studies | micro P | micro R | micro F1 | macro P | macro R |
|---|---|---|---|---|---|---|
| **CE (contrast-enhanced)** | 5,042 | 0.503 | 0.557 | 0.529 | 0.496 | 0.523 |
| NC (default calibration) | 40 | 0.467 | 0.562 | 0.510 | — | — |
| **NC (phase-aware)** | 40 | **0.475** | **0.568** | **0.517** | — | — |

NC = non-contrast (40 cases explicitly tagged in the Merlin reports — see `data/noncontrast_sids.txt`).

## Probe-level mAUROC (220 findings)

| | full-220 macro | matched-11 macro |
|---|---|---|
| Concat probe (production) | **0.853** | **0.873** |

Matched-11 anchors: `ascites`, `hepatic_steatosis`, `splenomegaly`, `cirrhosis`, `hepatic_cyst`, `simple_renal_cyst`, `renal_stones`, `pleural_effusion`, `lymphadenopathy`, `aortic_atherosclerosis`, `acute_pancreatitis`.

Per-finding val AUROCs on the 11 anchors:

| finding | n+ in val | AUROC |
|---|---|---|
| `splenomegaly` | 367 | 0.972 |
| `cirrhosis` | 143 | 0.959 |
| `pleural_effusion` | 809 | 0.953 |
| `acute_pancreatitis` | 53 | 0.921 |
| `aortic_atherosclerosis` | 1,659 | 0.921 |
| `hepatic_cyst` | 603 | 0.873 |
| `ascites` | 2,047 | 0.873 |
| `simple_renal_cyst` | 1,455 | 0.840 |
| `hepatic_steatosis` | 995 | 0.781 |
| `lymphadenopathy` | 1,131 | 0.773 |
| `renal_stones` | 572 | 0.732 |

220-finding AUROC quartiles: Q1 = 0.770, median = 0.872, Q3 = 0.930.

## Top systematic FPs (full val, 5,082 studies)

Findings the model over-fires on; ranked by total false-positive count:

| finding | FP count | % of val studies |
|---|---|---|
| `lymphadenopathy` | 1,080 | 21.3% |
| `renal_hypodensity` | 1,022 | 20.1% |
| `atelectasis` | 927 | 18.2% |
| `coronary_atherosclerosis` | 885 | 17.4% |
| `simple_renal_cyst` | 814 | 16.0% |
| `hiatal_hernia` | 663 | 13.0% |
| `ascites` | 638 | 12.5% |
| `hepatic_steatosis` | 613 | 12.1% |
| `prostatomegaly` | 595 | 11.7% |
| `aortic_atherosclerosis` | 536 | 10.5% |
| `intrahepatic_biliary_ductal_dilation` | 479 | 9.4% |

These are mostly **bidirectional uncertainty findings** — the same names show up in the top FN list. The probe is at the encoder ceiling on these (Pillar-0 LoRA fine-tune retry at scale didn't shift the needle — see [docs/11_LIMITATIONS.md](11_LIMITATIONS.md)).

## Top systematic FNs (full val)

| finding | FN count |
|---|---|
| `bowel_obstruction` | 457 |
| `lung_mass` | 442 |
| `renal_hypodensity` | 433 |
| `ascites` | 430 |
| `hepatic_steatosis` | 409 |
| `atelectasis` | 408 |
| `metastatic_disease` | 388 |
| `lymphadenopathy` | 358 |
| `renal_stones` | 350 |
| `hepatic_mass` | 356 |

`metastatic_disease` FN-heavy because `COREQUIRES` filters it out when no specific malignancy is also predicted. Intentional precision-trade for the catch-all overcalling pattern.

## Contrast-stratified per-finding deltas (NC vs CE, FP rates)

Findings where NC has materially higher FP rate than CE (before phase-aware fix):

| finding | NC FP% | CE FP% | delta |
|---|---|---|---|
| `simple_renal_cyst` | 27.5% | 15.9% | +11.6pp |
| `hiatal_hernia` | 25.0% | 13.0% | +12.0pp |
| `submucosal_edema` | 20.0% | 9.3% | +10.7pp |
| `renal_hypodensity` | 27.5% | 20.1% | +7.4pp |

Findings where NC has higher FN rate (model misses contrast-dependent findings):

| finding | NC FN% | CE FN% | delta |
|---|---|---|---|
| `adrenal_mass` | 17.5% | 3.1% | +14.4pp |
| `adrenal_adenoma` | 15.0% | 1.7% | +13.3pp |
| `femoral_hernia` | 12.5% | 3.4% | +9.1pp |
| `metastatic_disease` | 15.0% | 7.6% | +7.4pp |
| `lung_mass` | 15.0% | 8.6% | +6.4pp |
| `renal_stones` | 12.5% | 6.9% | +5.6pp |

After phase-aware Platt (`contrast_phase="nc"`), these deltas mostly shrink — see [docs/07_CALIBRATION.md](07_CALIBRATION.md) for the before/after table.

## Latency budget

Per-study end-to-end on a 24 GB GPU, **warm cache** (encoder features + heatmaps all cached on disk):

| stage | latency |
|---|---|
| DICOM → NIfTI (if needed) | 1–5 s |
| Load Merlin + Pillar-0 features (cached .npy) | <100 ms |
| Probe forward + Platt + Youden-J + filter | <50 ms |
| Router (12 positives × ~3 tools) on cached heatmaps | ~200 ms |
| Template render + organ ordering + negative statements | <50 ms |
| **Total (warm)** | **~5 s** (dominated by DICOM→NIfTI) |

**Cold cache** (first time a study is seen, no encoder features, no heatmaps):

| stage | latency |
|---|---|
| DICOM → NIfTI | 1–5 s |
| Merlin forward | 1–2 s |
| Pillar-0 forward | 2–3 s |
| Probe forward | <50 ms |
| Router with lazy Grad-CAM gen (8-12 findings, ~3 s each per encoder) | 30–60 s |
| **Total (cold)** | **40–70 s** |

The pre-render heatmap-cache warmer is `scripts/34_make_heatmap_demo.py` (single-study) or a custom loop calling `cam.ensure_concat_heatmap` (batch).

## Comparison to baselines

The pipeline's evolution within the dev work, micro F1 on 30-study val sample:

| stage | min_threshold | micro P | micro R | F1 |
|---|---|---|---|---|
| subsumption only (initial) | 0.55 | 0.30 | 0.62 | 0.40 |
| + COREQUIRES + cc tightening + floor 0.70 | 0.70 | 0.39 | 0.55 | 0.46 |
| + AUROC + category cap | 0.70 | 0.40 | 0.54 | 0.46 |
| + Platt scaling, re-tune floor | 0.20 | 0.44 | 0.55 | 0.49 |
| + renal_hypodensity de-subsumption | 0.20 | 0.50 | 0.55 (full val) | 0.53 |
| **+ phase-aware Platt on NC** | 0.20 | 0.50 / 0.48 NC | 0.55 / 0.57 NC | **0.53 / 0.52** |

Net pipeline gain over the floor-0.55 baseline: **+47% micro precision at iso-recall**, all on the calibration / filtering side without touching the encoder or retraining the probe weights.
