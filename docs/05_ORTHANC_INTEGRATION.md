# 05 — Orthanc integration

The model runs on a **GPU worker** (separate machine or container from Orthanc). Orthanc's job is to:

1. Receive DICOM studies (already established).
2. Detect newly-complete abdomen-pelvis CT studies.
3. Push a job to the broker queue with enough metadata for the worker to pick up the right DICOM series.

The worker does DICOM → NIfTI conversion itself; Orthanc does **not** need to serve NIfTI.

## Orthanc-side hook: study-completion trigger

Use either Lua (built-in) or the Python plugin. The Python plugin is preferred — it lets you reuse the DICOM-tag extraction helper from `deploy/example_orthanc_hook.py`.

### Detection criteria

A study qualifies if every series-level instance satisfies all of:

| DICOM tag | required value | rationale |
|---|---|---|
| `Modality` | `CT` | obvious |
| `BodyPartExamined` | `ABDOMEN`, `PELVIS`, `ABDOMENPELVIS`, `ABDOMEN_PELVIS`, or contains "ABD" | training was abdomen-pelvis CT only |
| `SliceThickness` | ≤ 5 mm | thinner = better; thicker = degraded performance |
| `ImageType` | does NOT contain `DERIVED`, `LOCALIZER`, `SCREEN_SAVE` | want primary axial reconstruction |

For multi-series studies (arterial + venous + delayed), pick the **venous / portal-venous phase** if present, otherwise the most-recent non-derived axial series.

### Contrast-phase detection

Read these tags from one representative instance of the chosen series:

```python
contrast_bolus_agent     = ds.get("ContrastBolusAgent")                  # (0018,0010)
contrast_bolus_volume    = float(ds.get("ContrastBolusVolume", 0) or 0)  # (0018,1041)
contrast_bolus_route     = ds.get("ContrastBolusRoute")                  # (0018,1040)
contrast_bolus_start_time = ds.get("ContrastBolusStartTime")             # (0018,1042)
```

Classification rule (battle-tested on Merlin):

```python
def classify_contrast_phase(ds) -> str:
    """Return 'ce' (contrast-enhanced), 'nc' (non-contrast), or 'unknown'."""
    agent  = (ds.get("ContrastBolusAgent") or "").strip()
    volume = float(ds.get("ContrastBolusVolume", 0) or 0)
    route  = (ds.get("ContrastBolusRoute") or "").strip().upper()

    # IV contrast administered → CE
    if agent and volume > 0 and route in {"", "IV", "INTRAVENOUS"}:
        return "ce"
    # Explicit non-contrast marker
    if not agent and volume == 0:
        # Cross-check with StudyDescription / SeriesDescription
        desc = " ".join(filter(None, [
            ds.get("StudyDescription", ""),
            ds.get("SeriesDescription", ""),
        ])).lower()
        if any(t in desc for t in ["non-contrast", "noncontrast", "without iv"]):
            return "nc"
        # If neither flagged and no explicit NC keyword: most likely CE per
        # reporting convention — but mark as 'unknown' to be safe.
        return "unknown"
    return "unknown"
```

**Recommendation**: when `unknown`, route the job as `ce` (matches the 99% training distribution) but tag the output with `contrast_phase_source: "default_ce"` so a reviewer can spot it.

## Job payload to the broker

Push to the broker (Redis, RabbitMQ, NATS, whatever your existing infra uses):

```json
{
  "study_id": "AC421363f",
  "orthanc_study_uuid": "f06f...",
  "modality": "CT",
  "body_part": "ABDOMENPELVIS",
  "slice_thickness_mm": 2.5,
  "contrast_phase": "ce",
  "contrast_phase_source": "ContrastBolusAgent+Volume",
  "dicom_dir": "/var/lib/orthanc/storage/f0/6f...",
  "series_uid_chosen": "1.2.840.113619...",
  "received_at": "2026-06-03T18:22:11Z"
}
```

The worker uses `dicom_dir` + `series_uid_chosen` to find the right series. If the worker can't reach Orthanc's storage directly, the hook should instead **download the DICOM series to a shared staging path** before pushing the job:

```python
# Pseudo-code in the Orthanc Python plugin
out_dir = f"/staging/{study_id}/"
os.makedirs(out_dir, exist_ok=True)
for instance_id in orthanc.RestApiGetJson(f"/series/{series_id}/instances"):
    dcm_bytes = orthanc.RestApiGet(f"/instances/{instance_id}/file")
    Path(f"{out_dir}/{instance_id}.dcm").write_bytes(dcm_bytes)
payload["dicom_dir"] = out_dir
broker.publish(payload)
```

## Returning results back to Orthanc

After the worker completes:

1. The worker writes the `FullReport` JSON to a results store of your choice (DB, S3, file).
2. The worker writes the per-finding heatmap NIfTIs to a path the viewer can serve.
3. Optional: write a DICOM-SR (Structured Report) for findings — Orthanc serves it like any other DICOM, and most PACS viewers display it inline. Sketch in `deploy/example_orthanc_hook.py:write_dicom_sr()`.

## A reference Orthanc hook

`deploy/example_orthanc_hook.py` is a runnable Python-plugin reference that:

1. Subscribes to `OnStableStudy` events
2. Filters by Modality + BodyPart + SliceThickness
3. Classifies contrast phase
4. Pushes a JSON job to Redis (you can swap the publisher for your broker)
5. Optional: writes a DICOM-SR after the worker returns

Adapt the publisher block to your existing broker SDK.

## Failures the hook should handle

- **Study with multiple non-derived series**: prefer venous → arterial → delayed; if no contrast tag present, prefer the latest series.
- **Foreign body / metal artifact studies**: out-of-distribution, performance unknown — recommend a `quality_flag` in the job payload that the worker can use to skip rather than report low-confidence findings.
- **Repeat studies for the same patient**: dedupe at the broker level (study_id is the obvious key).
