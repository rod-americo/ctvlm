# 09 — Grad-CAM heatmaps

Every Tier A/B sentence that includes an axial slice number, segment, side, or component-extent measurement is backed by a **per-encoder Grad-CAM heatmap** stored as a NIfTI aligned to the CT.

## How Grad-CAM is computed

`src/explain/cam.py` runs manual Grad-CAM (Selvaraju et al. 2017) via forward/backward hooks. The Captum path was abandoned because:

- Merlin uses `torch.utils.checkpoint` internally — Captum's captured activation gets detached from the autograd graph
- Pillar-0's `maxpool` is reachable via dotted attribute access but `nn.Module.__setattr__` rejects assigning a plain callable for monkey-patching

The hand-rolled hook-based recipe avoids both:

1. Register forward hook on target layer → capture activation `A`
2. Register `full_backward_hook` on target layer → capture gradient `dL/dA`
3. Spatial average pool the gradient → channel weights `w_c`
4. `cam = ReLU(sum_c w_c · A_c)` → upsample to CT resolution

### Per-encoder slot routing for the concat probe

The production probe is a single linear head `(220, 3200)`. To get a per-encoder Grad-CAM for finding `f`:

```python
W = probe.state_dict["weight"].numpy()    # (220, 3200)
W_M = W[:, :2048]                          # Merlin slot
W_P = W[:, 2048:]                          # Pillar-0 slot
b   = probe.state_dict["bias"].numpy()

# Merlin CAM for finding f: use W_M[f] as the linear head's row,
# applied AFTER per-encoder L2-normalisation (matches inference forward).
```

The L2-norm wrap matters: at inference the model computes `logits = W_M · L2(merlin) + W_P · L2(pillar0) + b`, so the gradient must flow through the L2 to get the right activation attribution. PyTorch handles the L2 Jacobian automatically when you compose the forward this way.

This is what `cam._merlin_cam_inner` / `cam._pillar0_cam_inner` do (with `apply_l2=True` for the concat probe path).

## Lazy generation

`cam.ensure_concat_heatmap(sid, encoder, finding)`:

1. Check `$CTVLM_WORK_ROOT/heatmaps/<encoder>/<sid>/<finding>.nii.gz`. If exists, return path immediately.
2. Otherwise load encoder + concat probe slot, run forward + backward, save NIfTI, return path.

Wall time per generation (24 GB GPU):

| | time | notes |
|---|---|---|
| Merlin Grad-CAM | ~3 s | dominated by encoder forward |
| Pillar-0 Grad-CAM | ~2 s | encoder forward + backward |

This is called automatically by `tools.cam_peak` and `tools.cam_connected_components` when the heatmap isn't cached. There's no other code path that triggers generation — the production worker just calls `pipeline.generate_report` and heatmaps materialise as a side effect.

## On-disk layout

```
$CTVLM_WORK_ROOT/heatmaps/
├── merlin/
│   ├── AC421363f/
│   │   ├── pleural_effusion.nii.gz       # 3D float16 [0..1]
│   │   ├── hepatic_cyst.nii.gz
│   │   └── ...
│   └── ...
└── pillar0/
    └── AC421363f/
        └── ...
```

Each NIfTI:
- Same shape and affine as the source CT
- Float16 values in [0, 1] (post-smoothing + percentile-clip + L1-normalize)
- A body mask (`HU > -500`) is applied so external-of-body voxels are forced to 0

Storage cost: ~5–15 MB per (study, finding, encoder). For 25k studies × ~5 findings × 2 encoders = ~1 TB at full coverage; in practice heatmap cache stays small because only positives generate.

## Serving heatmaps to the viewer

The reference NiiVue overlay URL format (the project ships a small file-server reference at `scripts/ct_files_server.py` though the production deployment can use any static-file server):

```
http://viewer/?
  ct=ct_volumes/AC421363f.nii.gz
  &masks=liver,spleen,kidney_left,kidney_right
  &heatmap=merlin:pleural_effusion,pillar0:pleural_effusion
  &slice=261
```

The viewer:
- Renders the CT volume
- Overlays the heatmap in a hot/cool colormap
- Jumps to axial slice 261 (the `cam_peak` slice from the recipe)
- Lets the radiologist toggle the Merlin vs Pillar-0 attribution

Two heatmaps side-by-side ("this is what each encoder thinks the finding is") makes the audit narrative compelling.

## Post-processing applied at save time

`cam.save_heatmap(h, aff, out_path, study_id=sid)`:

1. Optional `take_abs` (default off — Grad-CAM is already ReLU'd)
2. Body mask multiplication (`HU > -500`) → zeros background
3. Gaussian smoothing (σ=2.0 voxels) to remove block artifacts
4. Percentile clipping to (1, 99) on non-zero voxels
5. Min-max normalise to [0, 1]
6. Save as fp32 NIfTI with the CT's affine

Tune by passing different kwargs to `save_heatmap` if your viewer expects different intensity ranges.

## When Grad-CAM fails

`ensure_concat_heatmap` catches all exceptions and returns `None`. Failure paths:

| failure | what happens downstream |
|---|---|
| CUDA OOM on backward | tool returns `{"valid": False, "reason": "heatmap not cached"}`, Tier B sentence renders without slice number |
| Encoder model not loaded | same |
| Finding not in probe | same; logs warning |
| CT file missing | same; logs error |

The pipeline degrades gracefully — a missing heatmap produces a slightly thinner sentence but never crashes a report.

## Pre-generating heatmaps offline

If you want zero latency on Grad-CAM for the radiologist viewer, run a batch pre-generator for all positives across the cohort:

```python
# Pseudo-code (write your own loop in production)
from src.agent import pipeline
from src.explain import cam

for sid in study_ids:
    report = pipeline.generate_report(sid, skip_llm=True)
    for sf in report.structured:
        for enc in ("merlin", "pillar0"):
            cam.ensure_concat_heatmap(sid, enc, sf.finding)
```

At ~3 s per CAM × 2 encoders × ~6 positives/study × 25k studies = ~24 GPU-days for full coverage. In practice, generate on demand from viewer clicks and accept the first-view ~3 s delay.
