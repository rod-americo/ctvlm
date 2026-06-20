# deploy/

Reference scripts the coding agent should adapt to wire ctvlm into the existing
Orthanc + GPU broker setup. Each file is **self-contained and runnable** — the
contracts are correct; the broker SDK calls and Orthanc plugin loading need to
be plumbed to your actual infrastructure.

For the Docker Compose deployment profile used on `thor`, see
[`deploy/thor/README.md`](thor/README.md).

| file | role | docs |
|---|---|---|
| `example_orthanc_hook.py` | Orthanc Python plugin: study-completion → DICOM-tag extraction → broker publish | [docs/05_ORTHANC_INTEGRATION.md](../docs/05_ORTHANC_INTEGRATION.md) |
| `example_worker.py` | GPU worker entrypoint: warmup → handle_job → healthcheck | [docs/06_GPU_BROKER_INTEGRATION.md](../docs/06_GPU_BROKER_INTEGRATION.md) |
| `example_dicom_to_nifti.py` | dcm2niix wrapper used by the worker before inference | [docs/04_DATA_PIPELINE.md](../docs/04_DATA_PIPELINE.md) §"Stage 2" |

## End-to-end smoke flow (no Orthanc, no broker)

```bash
# 1. Pretend a hook fired; manually publish a job to stdin
echo '{
  "study_id": "AC421363f",
  "orthanc_study_uuid": "test",
  "modality": "CT",
  "contrast_phase": "ce",
  "dicom_dir": "/tmp/AC421363f_dicom/"
}' | python deploy/example_worker.py

# Output: one JSON status line on stdout with the FullReport summary.
```

For wiring to a real broker, replace the `for line in sys.stdin` loop in
`example_worker.py:main()` with your broker SDK's consumer pattern.

## Production checklist

Before going live, the coding agent should ensure:

- [ ] All `CTVLM_*` environment variables (see [docs/02_INSTALLATION.md](../docs/02_INSTALLATION.md)) are set on the worker
- [ ] `concat_rate_probe.pt` is on disk at `$CTVLM_CHECKPOINTS_DIR/concat_rate_probe.pt`
- [ ] `HF_TOKEN` is set, and the token's account has accepted the Merlin + Pillar-0 model gates on HuggingFace
- [ ] `dcm2niix` is on PATH (or fallback `pydicom` is installed)
- [ ] The worker has 24 GB GPU (or accept 5–10% F1 hit + serial encoder forwards on 16 GB; see [docs/06](../docs/06_GPU_BROKER_INTEGRATION.md))
- [ ] The Orthanc hook's broker publish (`publish_to_broker`) points at your real queue
- [ ] The worker's stdin loop is replaced with your real broker consumer
- [ ] Healthcheck (`/healthz`) is exposed for the orchestrator
- [ ] Metrics endpoint (`/metrics`) is exposed for Prometheus (see [docs/12](../docs/12_OPERATIONS.md))
- [ ] Disk-trim cron is configured (CT NIfTI + heatmaps + agent_cache)
- [ ] Worker rolling-restart policy is configured (every 6–12 h or every 500 jobs)
- [ ] One end-to-end smoke study has been run and the resulting FINDINGS:
      block has been reviewed by a radiologist
