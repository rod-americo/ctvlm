# 01 — System overview

## What this is

A deterministic radiology report generator for **CT abdomen-pelvis** studies. Takes a study ID, returns a structured FINDINGS block organised by anatomical section, every sentence traceable to a tool result.

The model is a **linear probe over a concatenated dual-encoder feature**:

```
CT volume ──► Merlin (2048-d global, L2-norm)  ──┐
                                                 ├──► 3200-d  ──► Linear  ──► 220 logits
CT volume ──► Pillar-0 (1152-d emb, L2-norm)  ──┘                 head
```

Probabilities go through a **deterministic agentic router** that calls tools (organ morphometrics, Grad-CAM peak/components, sub-organ localisation, HU sampling) per finding, then renders templated sentences with measurements.

## Why no LLM in the production path

The prose layer (MedGemma-4B + LoRA) is in the codebase for research but **disabled in deployment**. Why:

- LLM hallucinates concrete measurements and prior-study references that don't exist in the structured input
- Template-only output is **deterministic and 100% auditable** — every measurement, side, slice, and HU value comes from a tool call
- Latency drops from ~5s to <1ms for the prose layer
- No GPU footprint, no nondeterministic outputs, no failure modes

If you need radiology-style natural prose, a post-render regex check that drops any LLM sentence introducing tokens not in the structured input is the recommended approach. Not shipped.

## Data flow

```
┌──────────────────┐  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Orthanc:        │  │  Worker:       │  │  Inference:      │  │  Router:       │  │  Render:       │
│  study completed │→ │  DICOM→NIfTI   │→ │  encoder→probe  │→ │  recipe→tools  │→ │  templates→txt │
│  hook fires      │  │  + contrast    │  │  + Platt + Youd │  │  → structured  │  │  + neg-organ   │
│                  │  │    phase tag   │  │  → 220 probs    │  │    findings    │  │    statements  │
└──────────────────┘  └────────────────┘  └──────────────────┘  └────────────────┘  └────────────────┘
        ↓                    ↓                     ↓                    ↓                    ↓
   Lua/Python          /mnt/work/ct/        Merlin .npy +          Per-finding tool        FINDINGS:
   webhook to          AC*.nii.gz           Pillar-0 .npy          traces + Grad-CAM       <prose>
   worker queue                             (cached)               heatmap NIfTIs          (JSON return)
```

## Component map (which file owns what)

| layer | file | what it does |
|---|---|---|
| Path config | `src/config.py` | YAML + env var overrides (`CTVLM_*`) |
| Merlin encoder | `src/embeddings/merlin.py` | Load + forward + global_2048 extraction |
| Pillar-0 encoder | `src/embeddings/pillar0.py` | Load (PEFT-compat patch included) + forward + emb extraction |
| CT loader | `src/data/merlinplus.py` | `load_ct(sid)`, `load_mask(sid, organ)`, canonical RAS handling |
| Organ morphometry | `src/data/roi_crops.py` | Volume / mean HU / extent calculations from masks |
| **Pipeline entry** | `src/agent/pipeline.py` | `predict()`, `generate_report()` |
| Schema | `src/agent/schema.py` | `FullReport`, `StructuredFinding`, `ToolCall`, `ToolResult` |
| Canonical names | `src/agent/canonical.py` | 225-question → 220-canonical mapping |
| Recipes | `src/agent/recipes.py` | Per-finding tool DSL (Tier A hand-written, Tier B/C generic) |
| Tools | `src/agent/tools.py` | Pure functions: organ_morphometrics, cam_peak, etc. |
| Router | `src/agent/router.py` | Recipe execution + tool result composition |
| Templates | `src/agent/templates.py` | Sentence format strings |
| Renderer | `src/agent/render.py` | Assemble organ-grouped summary, filter (COREQUIRES/SUBSUMPTION/cap), negative statements |
| Cache | `src/agent/cache.py` | Per-study disk cache keyed by `(sid, tool, args_hash)` |
| Grad-CAM | `src/explain/cam.py` | Per-encoder Grad-CAM via concat probe weight slots, lazy `ensure_concat_heatmap()` |

## Files OUTSIDE the production path (research-only)

- `src/llm/`, `src/graph/`, `src/ontology/`, `src/retrieval/`, `src/segmentation/` — historical / deprecated branches. Safe to ignore for deployment.
- `scripts/41_merlin_rate_probe.py` — probe trainer; ship for re-calibration only
- `scripts/43_pillar0_lora.py` — encoder LoRA experiment, didn't pay off at scale
- `src/agent/render.render_prose` and `src/agent/render._load` — LLM call sites; never invoked in production

## Future work notes

- [CAC-DRS Chest CT probe](14_CACDRS_CHESTCT_PROBE.md) — research plan for training
  a CAC-DRS / Agatston probe on gated calcium-score CT labels using Pillar-0 ChestCT,
  then validating transfer to routine chest CT.
