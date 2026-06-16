# 06 — GPU broker integration

The worker is a Python process the existing GPU broker spawns / dispatches jobs to. This doc covers the **worker entrypoint contract** — what the worker function looks like, model lifecycle, error handling.

A reference implementation is in `deploy/example_worker.py`.

## Worker entrypoint contract

```python
def handle_job(job: dict) -> dict:
    """
    Inputs: the JSON payload published by the Orthanc hook (see docs/05).
    Returns: a JSON-serialisable dict with status + report or error metadata.
    Side effects: writes encoder features, heatmaps, agent cache, and the
                  FullReport JSON under CTVLM_WORK_ROOT.
    """
    ...
```

The broker is expected to:
- Dispatch one job at a time per worker process (the model singletons are NOT thread-safe).
- Pass a soft timeout (recommended ≥ 90 s; cold-cache studies with many positives can exceed 60 s).
- Kill + restart the worker if RSS exceeds 22 GB or the GPU OOMs unrecoverably.

## Model lifecycle

The worker should **load all models at startup** (before announcing readiness to the broker), so the first job doesn't pay the model-load tax:

```python
# Module-scope at worker start
from src.embeddings import merlin, pillar0
from src.agent import pipeline

print("loading Merlin ...");  merlin.load_model()             # ~5 s
print("loading Pillar-0 ..."); pillar0.load_model()           # ~3 s
print("warming probe cache..."); pipeline._load_probe(pipeline.DEFAULT_PROBE)
print("READY")
```

After that, jobs are sub-second to several seconds depending on cache state.

## Job processing pseudo-code

```python
import os, json, time, traceback, dataclasses
from pathlib import Path
from src.agent import pipeline

WORK_ROOT = Path(os.environ["CTVLM_WORK_ROOT"])
CT_DIR    = WORK_ROOT / "ct_volumes"
REPORTS   = WORK_ROOT / "reports_out"

def handle_job(job: dict) -> dict:
    sid = job["study_id"]
    phase = job.get("contrast_phase", "ce")
    if phase == "unknown":
        phase = "ce"   # see docs/05 — default fallback for routing

    t0 = time.time()
    try:
        # 1. DICOM → NIfTI (if not already present)
        nifti_path = CT_DIR / f"{sid}.nii.gz"
        if not nifti_path.exists():
            from deploy.example_dicom_to_nifti import convert
            convert(job["dicom_dir"], CT_DIR, sid)

        # 2. Inference
        report = pipeline.generate_report(
            sid,
            skip_llm=True,
            contrast_phase=phase,
            max_findings=12,
        )

        # 3. Persist
        out_json = REPORTS / f"{sid}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(dataclasses.asdict(report), indent=2))

        return {
            "status": "ok",
            "study_id": sid,
            "report_path": str(out_json),
            "n_positives": len(report.positives),
            "latency_s": time.time() - t0,
            "encoder": report.encoder,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"status": "retry", "study_id": sid, "error": "cuda_oom",
                "detail": str(e)[:200]}
    except Exception as e:
        return {"status": "error", "study_id": sid,
                "error": type(e).__name__,
                "detail": str(e)[:500],
                "traceback": traceback.format_exc(limit=10)}
```

## Status taxonomy

The worker should always return one of:

| status | meaning | broker action |
|---|---|---|
| `ok` | report written | mark job complete |
| `retry` | transient (OOM, lock contention) | requeue with backoff |
| `error` | logic error, bad input, unrecoverable | dead-letter |
| `reject` | study failed pre-checks (wrong modality, missing series) | mark complete with reject |

## GPU memory budget per job

Steady state during inference on a 24 GB GPU:

| | bytes |
|---|---|
| Merlin model | ~1.0 GB |
| Pillar-0 model | ~0.7 GB |
| Concat probe + Platt arrays | < 5 MB |
| Merlin forward activations (with grad checkpointing) | ~6 GB |
| Pillar-0 forward + 384³ buffers | ~9 GB |
| Grad-CAM backward pass (one finding at a time) | ~3 GB peak |
| Misc CUDA caching | ~1 GB |

The two encoders **cannot both be holding peak activations at the same time**. The worker should run Merlin → release → Pillar-0 → release → Grad-CAM. The pipeline already does this in its forward path (each encoder's forward is in its own scope; activations drop out of scope before the next encoder runs).

If your GPU is < 24 GB:

```python
# Force serial encoder forwards + aggressive cache clear
import torch
torch.cuda.empty_cache()
merlin_feat = merlin_forward(...)
torch.cuda.empty_cache()
pillar0_feat = pillar0_forward(...)
torch.cuda.empty_cache()
```

This adds ~1 s per study but keeps the peak GPU footprint under ~16 GB.

## Multi-GPU scaling (horizontal)

To process more studies per second:

1. **One worker process per GPU**. Each process loads its own Merlin + Pillar-0 (the load tax is paid once at startup).
2. **Broker round-robins or load-balances** jobs across workers.
3. **Shared disk** for features + heatmaps. The `merlin_global/`, `pillar0_emb/`, `heatmaps/`, `agent_cache/` directories MUST be a shared mount (NFS, EFS, etc.) so cache hits across workers actually hit.

There is **no benefit to data-parallel on a single study** — the encoders are single-stream and the bottleneck is preprocessing + I/O, not GPU compute.

## Health checks

The worker should expose a readiness probe the broker can hit:

```python
def healthcheck() -> dict:
    """Return status + last successful job timestamp.

    Includes a 'model_loaded' flag the broker can poll at startup.
    """
    return {
        "model_loaded": "merlin" in pipeline._probe_cache or pillar0._model_loaded(),
        "last_ok_ts": getattr(handle_job, "_last_ok_ts", None),
        "gpu_free_gb": torch.cuda.mem_get_info()[0] / 1e9,
    }
```

## Recommended worker config

| setting | value | rationale |
|---|---|---|
| `--max-jobs-per-worker` | 500 | cycle workers periodically to avoid CUDA memory fragmentation |
| `--soft-timeout` | 90 s | covers cold-cache + 12-positive study |
| `--hard-timeout` | 180 s | runaway protection |
| `--prefetch` | 1 | one job in flight, one queued |
| `--start-delay` | 30 s | give model load time before first job |
| restart policy | `on-failure` | with exponential backoff |
