# 12 — Operations runbook

## Logging contract

Every job should emit a structured log line with these fields:

```json
{
  "ts": "2026-06-03T18:22:11Z",
  "level": "INFO",
  "event": "job_complete",
  "study_id": "AC421363f",
  "contrast_phase": "ce",
  "n_positives": 8,
  "n_negative_statements": 4,
  "encoder": "merlin+pillar0_L2",
  "probe_path": "/opt/ctvlm/models/concat_rate_probe.pt",
  "latency_total_s": 5.2,
  "latency_dicom_to_nifti_s": 2.1,
  "latency_encoder_forward_s": 0.0,
  "latency_probe_and_calibration_s": 0.04,
  "latency_router_s": 0.18,
  "latency_render_s": 0.02,
  "cache_hits": { "merlin_feature": true, "pillar0_feature": true, "heatmaps": 7 },
  "cache_misses": { "heatmaps": 1 },
  "positives": ["pleural_effusion", "cardiomegaly", ...]
}
```

Failure path:

```json
{
  "ts": "...",
  "level": "ERROR",
  "event": "job_failed",
  "study_id": "AC421363f",
  "error": "OutOfMemoryError",
  "stage": "encoder_forward",
  "detail": "CUDA out of memory. Tried to allocate ...",
  "gpu_free_gb_at_fail": 0.4
}
```

Ship to your existing log aggregator (ELK, Datadog, Splunk). The `event` field is the primary index.

## Metrics to scrape

Expose Prometheus-style metrics on a metrics port:

| metric | type | labels |
|---|---|---|
| `ctvlm_jobs_total` | counter | `status` ∈ {ok, error, retry, reject} |
| `ctvlm_job_latency_seconds` | histogram | `phase`, `cache_state` ∈ {cold, warm} |
| `ctvlm_positives_per_study` | histogram | `contrast_phase` |
| `ctvlm_encoder_forward_seconds` | histogram | `encoder` ∈ {merlin, pillar0} |
| `ctvlm_gradcam_seconds` | histogram | `encoder`, `finding` (cap to top-30 by volume) |
| `ctvlm_gpu_free_bytes` | gauge | — |
| `ctvlm_cache_hit_ratio` | gauge | `kind` ∈ {merlin_feature, pillar0_feature, heatmap, tool_result} |
| `ctvlm_finding_predicted_total` | counter | `finding` (one series per canonical name) |
| `ctvlm_model_load_age_seconds` | gauge | — (time since the worker loaded its models; cycle if too old) |

## Alerts

| alert | condition | action |
|---|---|---|
| HighFailureRate | `rate(ctvlm_jobs_total{status=~"error\|retry"}[15m]) > 0.05` | page on-call |
| HighOOMRate | `rate(ctvlm_jobs_total{status="retry",error="cuda_oom"}[15m]) > 0.02` | check GPU contention / restart worker |
| StalledJob | `rate(ctvlm_jobs_total{status="ok"}[5m]) == 0` AND broker queue non-empty | check worker process / GPU driver |
| CacheHitDropping | `ctvlm_cache_hit_ratio{kind="merlin_feature"} < 0.5` over 30m | likely disk filled or NFS issue |
| FindingDrift | `rate(ctvlm_finding_predicted_total{finding="X"}[6h])` deviates > 3× from 30d baseline | calibration drift on that finding |
| ModelStale | `ctvlm_model_load_age_seconds > 24h` | rolling-restart the worker pool |

## Disk hygiene

```
$CTVLM_WORK_ROOT/
  ct_volumes/        keep   N days (size: large)
  merlin_global/     keep   forever (size: small — 2 KB/study)
  pillar0_emb/       keep   forever (size: small — 2 KB/study)
  heatmaps/          keep   N days (size: large — ~5 MB/(study,finding,encoder))
  agent_cache/       keep   N days (size: small)
  reports_out/       archive to long-term store
```

Suggested cron:

```cron
# Trim heatmaps older than 30 days
0 2 * * * find $CTVLM_WORK_ROOT/heatmaps -name '*.nii.gz' -mtime +30 -delete

# Trim CT NIfTI staging older than 7 days (Orthanc keeps DICOM, we don't need the converted NIfTI long)
0 3 * * * find $CTVLM_WORK_ROOT/ct_volumes -name '*.nii.gz' -mtime +7 -delete

# Trim agent_cache (tool results) older than 30 days
0 4 * * * find $CTVLM_WORK_ROOT/agent_cache -name '*.json' -mtime +30 -delete
```

Encoder features (`merlin_global/`, `pillar0_emb/`) **should not be trimmed** — they're tiny (2 KB/study) and re-extraction is expensive. They're the production cache of CT → features.

## Cycle / rolling restart

CUDA memory fragmentation accumulates over hours. Cycle workers every 6–12 hours or every N jobs (whichever first):

```bash
# Worker self-exit after 500 jobs; broker respawns
if [ "$JOBS_PROCESSED" -gt 500 ]; then
    exit 0
fi
```

The 30 s startup cost (model loads) is paid once per cycle. With a multi-worker pool (3+), staggered restarts keep capacity steady.

## Health endpoint

The worker exposes `GET /healthz` returning:

```json
{
  "status": "ready" | "loading" | "draining",
  "model_loaded": true,
  "last_ok_ts": "2026-06-03T18:22:11Z",
  "last_error_ts": "2026-06-03T17:45:09Z",
  "gpu_free_gb": 12.3,
  "uptime_s": 14523,
  "jobs_processed": 432
}
```

Broker polls before dispatching; orchestrator polls for liveness.

## Probe checkpoint rollout

The probe is a single 3 MB file; rollout is fast and reversible:

```bash
# Stage new probe
scp concat_rate_probe.pt worker:/opt/ctvlm/models/concat_rate_probe.pt.new

# Atomic swap, rolling restart workers
ssh worker "mv /opt/ctvlm/models/concat_rate_probe.pt.new /opt/ctvlm/models/concat_rate_probe.pt"
ssh worker "systemctl restart ctvlm-worker@*"

# Rollback if needed
ssh worker "mv /opt/ctvlm/models/concat_rate_probe.pt.prev /opt/ctvlm/models/concat_rate_probe.pt"
```

Always keep the previous probe as `.prev` for one-command rollback. Compare per-finding mAUROC on a regression cohort before fully rolling out.

## Diagnostic / debugging session

When a single study's report looks wrong:

```bash
# Single-study demo with full trace, no LLM
python scripts/39_agentic_report_demo.py --sid <study_id> --max-findings 20 --skip-llm

# Re-run with explicit non-contrast calibration
python -c "
from src.agent import pipeline
import dataclasses, json
r = pipeline.generate_report('<study_id>', skip_llm=True, contrast_phase='nc')
print(json.dumps(dataclasses.asdict(r), indent=2, default=str))
"
```

The output includes every tool call's args + result + latency + cache hit status — sufficient to diagnose where a finding came from.
