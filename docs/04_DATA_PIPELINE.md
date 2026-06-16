# 04 — Data pipeline (DICOM → report)

End-to-end stages a study traverses. Latency numbers are typical on a 24 GB GPU; cold-cache (first time a study is seen) is much slower than warm-cache (heatmaps + features cached on disk).

```
┌────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐
│  ① Intake  │→ │ ② DICOM→NIfTI│→ │ ③ Encoder forward│→ │ ④ Probe + cal  │→ │ ⑤ Router → text│
└────────────┘  └──────────────┘  └──────────────────┘  └────────────────┘  └────────────────┘
   from           write to            cache features        Platt+Youden-J      Tier A/B/C
   Orthanc        ct_volumes/         to disk               + filters           template render
```

## Stage 1 — Intake (Orthanc-side)

The Orthanc server fires a study-completion hook (Lua or Python). The hook should:

1. Filter to **abdomen-pelvis CT studies only** — check `BodyPartExamined` ∈ {`ABDOMEN`, `PELVIS`, `ABDOMENPELVIS`} or equivalent and `Modality == 'CT'`.
2. Extract contrast phase from `ContrastBolusAgent` / `ContrastBolusVolume` (see [docs/05_ORTHANC_INTEGRATION.md](05_ORTHANC_INTEGRATION.md)).
3. Push a job to the GPU broker:
   ```json
   {
     "study_id": "AC421363f",
     "orthanc_study_uuid": "<orthanc uuid>",
     "contrast_phase": "ce" | "nc" | "unknown",
     "dicom_dir": "/orthanc/storage/<uuid>"
   }
   ```

Output of this stage: a job message on the broker queue.

## Stage 2 — DICOM → NIfTI

The worker picks up the job. Before any encoder forward, the DICOM series must become a **canonical-RAS NIfTI**. Use `dcm2niix` (battle-tested) or `nibabel`:

```bash
dcm2niix -f "%a" -o /opt/ctvlm/work/ct_volumes/ -z y /orthanc/storage/<uuid>
# produces AC<study_id>.nii.gz
```

Then the encoders' preprocess paths will reorient via `nib.as_closest_canonical()`.

**Important: do NOT pre-resize to 384³ here.** Each encoder has its own resize (Merlin: ~3 mm spacing, Pillar-0: 384³). Preprocessing is owned by each encoder's loader.

Latency: 1–5 s depending on series size + disk speed.

Output: `/opt/ctvlm/work/ct_volumes/AC<sid>.nii.gz`

## Stage 3 — Encoder forwards

Both encoders run on the same CT (parallel-loadable, sequential-friendly):

| | Merlin global_2048 | Pillar-0 emb_1152 |
|---|---|---|
| Input shape | (1, 1, S, A, R) at 1.5×1.5×3 mm | (1, 11, 384, 384, 384) — 11 CT windows |
| Forward time | 0.5 – 1.5 s | 2 – 3 s |
| Output | (2048,) fp16 → `merlin_global/AC<sid>.npy` | (1152,) fp16 → `pillar0_emb/AC<sid>.npy` |
| Model entrypoint | `src.embeddings.merlin.load_model()` | `src.embeddings.pillar0.load_model()` |
| Cache behaviour | Read .npy if exists, else run forward + save | Same |

If the .npy file exists on disk, **the encoder forward is skipped entirely** — features are loaded directly. The first request for a new study pays the full 3-4 s; subsequent calls are sub-100 ms.

## Stage 4 — Probe + calibration

`src.agent.pipeline.predict(sid)`:

1. Load Merlin .npy + Pillar-0 .npy
2. L2-normalise each, concatenate → 3200-d feature
3. Forward through the linear probe → raw logits (220,)
4. Apply per-finding Platt: `cal_logits = A * raw + B`  (A and B are per-finding, NC-aware if `contrast_phase="nc"`)
5. Sigmoid → probabilities

Latency: <50 ms (pure numpy after the features are loaded).

## Stage 5 — Router + render

For each finding above its effective threshold:

1. **Recipe lookup** — `src.agent.recipes.lookup(canonical)` returns a `Recipe` (Tier A hand-written / Tier B organ-anchored / Tier C generic fallback).
2. **Tool execution** — `src.agent.router.route(sid, finding, prob)` runs each `ToolCall` through the cache:
   - Most expensive tools: `cam_peak` and `cam_connected_components` — both trigger lazy `ensure_concat_heatmap()` if no NIfTI is cached for `(sid, encoder, finding)`. Generating one Merlin Grad-CAM is ~3 s; Pillar-0 is ~2 s.
   - Cheap tools (organ mask reads, HU sampling): <100 ms each.
3. **Template render** — `src.agent.templates.render_finding(sf)` fills the chosen template string with tool outputs.
4. **Assembly** — `src.agent.render.assemble_summary(structured, probs)`:
   - Subsumption + COREQUIRES + per-category cap (see [docs/07_CALIBRATION.md](07_CALIBRATION.md))
   - Negative-organ statements ("Liver: normal, no focal lesion") when an organ category has no positive AND max prob in it is below `negative_threshold=0.40`
   - Anatomical-section ordering (Thorax → Liver → ... → Musculoskeletal → Multi-organ)

Latency:
- Cold case (heatmaps cold): 30–60 s for 8-12 positives (each unique CAM is ~3 s × 2 encoders if router asks for both)
- Warm case (heatmaps cached): <500 ms

## Output payload

The worker returns a `FullReport` dataclass (`asdict()` for JSON). Persist `summary_text`, `positives`, and the per-finding `structured` entries (with tool traces) into your existing reporting system / database. The heatmap NIfTIs stay on disk and can be served on demand to the radiologist's viewer (see [docs/09_GRAD_CAM.md](09_GRAD_CAM.md)).

## Failure modes & fallbacks per stage

| stage | failure | fallback |
|---|---|---|
| 1 — Intake | non-abdomen study, contrast unknown | reject at hook; alert ops queue |
| 2 — DICOM→NIfTI | dcm2niix fails (multi-frame, bad header) | log + reject; alert |
| 3 — Encoder forward | OOM | retry once with `torch.cuda.empty_cache()`; if still OOM, alert |
| 3 — Encoder forward | model load fails (HF token expired) | fail entire job; alert |
| 4 — Probe | feature file missing | re-run encoder forward (recover from disk-write race) |
| 5 — Router | tool exception | tool returns `{"error": ...}` in trace; template renders without that field |
| 5 — Grad-CAM gen | OOM in backward pass | catch, return `{"valid": False, "reason": "..."}`, Tier B sentence renders without slice |
| 5 — Empty positives | model surfaced 0 positives at threshold | return summary with only negative-organ statements + "No acute findings on this study." |
