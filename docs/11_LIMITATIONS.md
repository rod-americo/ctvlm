# 11 — Limitations & known issues

## Regulatory status

**This is a research-grade pipeline. NOT FDA-cleared, NOT CE-marked.** Do not use as a sole basis for clinical decisions. The output is intended as a draft / triage / quality-check input to a qualified radiologist, never as the final report.

The deployment doc should reflect this everywhere the radiologist sees a generated report.

## Training distribution & generalisation

### What it was trained on

- **Merlin Abdominal CT dataset** (Stanford), 25,494 reports total → 25,414 cases with both encoders cached + RATE labels
- Imaging: predominantly axial-reconstructed abdomen-pelvis CT, 2.5–5 mm slice thickness
- Demographics: single-institution US tertiary centre; expect distribution drift on other populations

### What it was NOT trained on

| out-of-distribution | implication |
|---|---|
| Chest CT | the "Visible Thoracic" category is from incidental upper-cuts; do NOT route chest CT through this pipeline |
| Pediatric studies | the cohort skews adult; rare pediatric findings have <10 positives |
| Very thin slices (<1.5 mm) or very thick (>5 mm) | encoders' preprocessing handles, but performance is unverified |
| Contrast phases other than venous / "routine portal venous" | ~99% of training is the routine portal-venous phase. Arterial-only or delayed-only studies may underperform on enhancement-dependent findings |
| **Non-contrast** | 1.1% of training is explicit NC. Phase-aware Platt fits only 26 of 220 findings. See [docs/07_CALIBRATION.md](07_CALIBRATION.md) |
| Post-operative / post-instrumentation | drains, stents, sutures may confuse some findings |
| Trauma protocols (whole-body, fast scan) | underrepresented |

### Don't claim what we didn't test

Per-finding AUROC ≠ per-population AUROC. The 0.872 matched-11 macro is on the held-out 20% from the **same Merlin distribution**. External validity (different scanner, different population, different protocol) is unverified.

## Known FP / FN patterns

See [docs/10_PERFORMANCE.md](10_PERFORMANCE.md) for the full list. The recurring patterns:

| pattern | example findings |
|---|---|
| Bidirectional uncertainty (in both top FPs and top FNs) | `ascites`, `atelectasis`, `hepatic_steatosis`, `lymphadenopathy` |
| Elderly-correlate over-firing | `aortic_atherosclerosis`, `coronary_atherosclerosis`, `coronary_artery_calcification` |
| Co-occurrence cluster | pancreatic cluster (`pancreatic_atrophy` + `pancreatic_duct_dilatation` + `IPMN` all fire together when one is real) |
| Male-pelvis over-firing | `prostatomegaly` on essentially every male case |
| Cyst confusion on non-contrast | `simple_renal_cyst` ↑ FP on NC because any hypodensity looks cyst-like without enhancement |

These are **encoder-ceiling** issues, not pipeline ones. Mitigations attempted:

- **Pillar-0 LoRA fine-tune (aggressive r=8, all attention)** — catastrophic forgetting (matched macro −0.066 at 1k cases). Documented in `agentic_pipeline_calibration` memory.
- **Pillar-0 LoRA fine-tune (conservative r=2, atlas_models.2 only)** — preserved base geometry (cosine 0.66 vs 0.475 for aggressive) and showed +0.020 matched-macro on 1,020 cases, but **washed out at 10,280 cases** (+0.003 full macro, matched tied). Not shipped.
- **MLP probe (3200→512→220)** — tied with linear probe within +0.004 macro mAUROC, no pipeline-level lift after Platt + Youden-J. Not shipped.

## Specific known bugs

| | description | workaround |
|---|---|---|
| `gallbladder_stones` size | The `cam_connected_components` extent at threshold 0.7 sometimes reports unrealistic sizes (10+ cm) because the CAM is spatially diffuse. | Either bump cc threshold to 0.85 in the recipe (recall drop), or suppress the size_clause from the template, or post-render regex filter sizes > 5 cm. |
| Cosmetic "no findings" message | If 0 positives clear thresholds, the renderer emits only negative-organ statements with no "no acute findings" sentence. | Add a guard at the top of `render.assemble_summary` if positives is empty. |
| `axial_slice` shown as voxel index | Some sentences report `(axial slice N)` where N is a voxel index, not a clinically-numbered slice. | Acceptable for triage; for radiologist-display, transform via known slice-count metadata. |

## Things to monitor in production

Set alerts on:

- **Average positives per study** drifts > 20% from baseline (8.5) — likely calibration drift
- **Median latency** drifts > 50% from baseline (~5 s warm / ~50 s cold) — likely Grad-CAM cache miss explosion or disk slowdown
- **Worker OOM rate** > 1% — likely GPU contention or model load corruption
- **NC contrast-phase fraction** drifts — if NC fraction climbs above ~5%, retrain phase Platt with the expanded NC pool
- **Specific findings with new high FP rate**: if a finding starts surfacing on >40% of studies, almost certainly a calibration bug — recompute Platt for that finding

See [docs/12_OPERATIONS.md](12_OPERATIONS.md) for the operations runbook.

## What we'd do next if we had time

Concrete EV-ordered list:

1. **Sigmoid temperature scaling per finding per phase** — current Platt is global. Phase-conditional + finding-conditional joint fit would shave another 0.01–0.02 F1, especially on NC.
2. **More non-contrast training cases** — even 500–1000 more NC cases would unlock NC Platt fits on most of the 220 findings.
3. **Per-finding co-occurrence model** — many of the top FPs are pure co-occurrence noise (pancreatic cluster, vascular cluster). A pairwise log-linear correction on the post-Platt probabilities would catch these without retraining.
4. **Encoder LoRA on 10k+ cases** — the 2k LoRA run plateaued without lift. With 10k+ training cases and a wider target scope, encoder-side gains might materialise. Cost: ~10 GPU-days for training + 20 hr for re-extraction.
5. **Multi-finding head** — replace the 220-output linear with a transformer that attends across findings. Could exploit co-occurrence priors structurally.
