# 02 — Installation

## Hardware requirements

| | minimum | recommended |
|---|---|---|
| GPU | 16 GB (single Merlin OR Pillar-0 forward at a time) | **24 GB** (Merlin + Pillar-0 simultaneous + Grad-CAM headroom) |
| RAM | 32 GB | 64 GB |
| CPU | 8 cores | 16 cores (parallel preprocessing) |
| Disk | 200 GB (encoder caches + heatmaps) | 1 TB (room for 25k+ studies + heatmap cache) |
| CUDA | 12.4 | 12.4 |

The production deployment was validated on a single RTX 3090 24 GB (Ampere sm_86). The 24 GB threshold matters because **Merlin + Pillar-0 forward + Grad-CAM backward pass through the concat probe** is the tightest moment — falling below 24 GB forces serial Merlin → Pillar-0 forwards with a `torch.cuda.empty_cache()` between, adding ~3 s per study.

## Python environment

```bash
# Create env
python3.13 -m venv /opt/ctvlm-venv
source /opt/ctvlm-venv/bin/activate

# Install with the inference deps
pip install --upgrade pip
pip install -e /path/to/ctvlm/repo

# If you need the (research-only) LLM prose layer too:
pip install -e "/path/to/ctvlm/repo[llm]"
```

## Required model assets

| asset | path / source | size | role |
|---|---|---|---|
| Merlin weights | `stanfordmimi/Merlin` (HF) — gated, accept license | ~500 MB | 3D ResNet-152 encoder for global_2048 features |
| Pillar-0 weights | `YalaLab/Pillar0-AbdomenCT` (HF) — gated | ~600 MB | Atlas abdomen-CT encoder for emb_1152 features |
| **Production probe** | `concat_rate_probe.pt` (ship with deployment) | ~3 MB | 3200→220 linear head + Platt + Youden-J + per-finding val AUROC + NC-specific calibration |
| RATE canonical map | `data/rate_canonical_map.csv` (in repo) | 7 KB | 225 RATE questions → 220 canonical snake_case names |
| Non-contrast SID list | `data/noncontrast_sids.txt` (in repo) | 2 KB | 190 study IDs explicitly tagged non-contrast in MerlinPlus reports — used by the trainer for phase-stratified Platt fit |

The probe checkpoint is the **only** trainable asset that needs to ship with the deployment. The encoders are downloaded from HF on first load (cached in `$HF_HOME`).

## Environment variables (production worker)

Every path-config field has a `CTVLM_*` env-var override (introduced in `src/config.py:load_paths`):

```bash
# Where ALL generated artefacts live (features, heatmaps, agent cache, logs)
export CTVLM_WORK_ROOT=/opt/ctvlm/work

# Where the probe checkpoint lives
export CTVLM_CHECKPOINTS_DIR=/opt/ctvlm/models

# HF cache for the encoders
export CTVLM_HF_CACHE=/opt/ctvlm/hf_cache
export HF_HOME=/opt/ctvlm/hf_cache             # also set this for `transformers`

# CT volumes (per-study .nii.gz) — input to the encoders
# These are the files DICOM→NIfTI conversion writes to
export CTVLM_MERLIN_ROOT=/opt/ctvlm/work/ct_volumes

# Optional: organ segmentation masks (only required for Tier A tools like
# liver_to_spleen_hu_ratio, organ_morphometrics, liver_segment_at, etc.)
export CTVLM_MERLIN_PLUS_DIR=/opt/ctvlm/work/organ_masks
export CTVLM_MASKS_DIR=/opt/ctvlm/work/organ_masks

# Logging
export CTVLM_LOGS_DIR=/opt/ctvlm/logs
export CTVLM_REPORTS_DIR=/opt/ctvlm/reports

# Required for first-time HF model download (gated repos)
export HF_TOKEN=<your-token-with-accepted-merlin-and-pillar0-license>
```

You can also keep a `configs/paths.yaml` and let env vars override individual fields. The env-var path is preferred in production because it's easier to manage with systemd / Docker.

## File-system layout the worker expects

```
/opt/ctvlm/
├── models/
│   └── concat_rate_probe.pt
├── work/
│   ├── ct_volumes/                         # per-study NIfTI (DICOM converted)
│   │   └── AC<study_id>.nii.gz
│   ├── organ_masks/                        # optional, for Tier A tools
│   │   └── AC<study_id>/
│   │       ├── liver.nii.gz
│   │       ├── spleen.nii.gz
│   │       └── ...
│   ├── merlin_global/                      # cached encoder features
│   │   └── AC<study_id>.npy                # 2048-d fp16
│   ├── pillar0_emb/
│   │   └── AC<study_id>.npy                # 1152-d fp16
│   ├── heatmaps/                           # Grad-CAM NIfTIs
│   │   ├── merlin/AC<sid>/<finding>.nii.gz
│   │   └── pillar0/AC<sid>/<finding>.nii.gz
│   ├── agent_cache/                        # tool result JSON cache
│   │   └── AC<sid>/<tool>__<hash>.json
│   └── hf_cache/                           # transformers cache
├── logs/
└── reports/
```

## Sanity check after install

```bash
python -c "
from src.agent import pipeline
import json, dataclasses
# Smoke-test that the probe + encoders load cleanly
probs, findings, encoder, thresholds, aucs = pipeline.predict('AC421363f')
print('OK — encoder:', encoder, 'n_findings:', len(findings))
print('top 5 probabilities:')
for k in sorted(probs, key=lambda x: -probs[x])[:5]:
    print(f'  {k}: {probs[k]:.3f}  thr={thresholds.get(k, 0.5):.3f}')
"
```

If this prints the top 5 findings and their thresholds, the pipeline is wired correctly. Else see [docs/12_OPERATIONS.md](12_OPERATIONS.md) "Troubleshooting".
