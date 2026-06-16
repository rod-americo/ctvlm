"""Reference worker entrypoint.

Adapt the broker integration (Redis, RabbitMQ, NATS, gRPC, whatever you have)
to your existing infrastructure. The `handle_job(job: dict) -> dict` contract
is what matters; everything else is glue.

The worker:
  1. Loads Merlin + Pillar-0 + concat probe at startup (before announcing readiness)
  2. Receives jobs from the broker (see deploy/example_orthanc_hook.py for the
     payload shape)
  3. Converts DICOM → NIfTI if needed (see deploy/example_dicom_to_nifti.py)
  4. Runs `pipeline.generate_report(sid, skip_llm=True, contrast_phase=...)`
  5. Writes the FullReport JSON to disk + returns status dict
  6. On OOM: empty cache, return retry. On other errors: dead-letter.

Run with:
    CTVLM_WORK_ROOT=/opt/ctvlm/work CTVLM_CHECKPOINTS_DIR=/opt/ctvlm/models \\
        python deploy/example_worker.py
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch                                                    # noqa: E402
from src.agent import pipeline                                  # noqa: E402

log = logging.getLogger("ctvlm.worker")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

WORK_ROOT = Path(os.environ.get("CTVLM_WORK_ROOT", "/opt/ctvlm/work"))
CT_DIR = WORK_ROOT / "ct_volumes"
REPORTS_DIR = WORK_ROOT / "reports_out"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── lifecycle ────────────────────────────────────────────────────────────── #

_state = {"last_ok_ts": None, "last_err_ts": None,
          "jobs_processed": 0, "model_loaded": False}


def warmup() -> None:
    """Load all models. Called once before the worker announces readiness."""
    log.info("loading Merlin ...")
    from src.embeddings import merlin
    merlin.load_model()
    log.info("loading Pillar-0 ...")
    from src.embeddings import pillar0
    pillar0.load_model()
    log.info("warming probe checkpoint cache ...")
    pipeline._load_probe(pipeline.DEFAULT_PROBE)
    _state["model_loaded"] = True
    log.info("worker READY  gpu_free_gb=%.1f", torch.cuda.mem_get_info()[0] / 1e9)


# ── job handler ──────────────────────────────────────────────────────────── #

def handle_job(job: dict) -> dict:
    """Process one inference job. See docs/06_GPU_BROKER_INTEGRATION.md."""
    sid = job["study_id"]
    phase = job.get("contrast_phase") or "ce"
    if phase == "unknown":
        # Per docs/05: default to CE (matches 99% of training distribution)
        phase = "ce"
        phase_source = job.get("contrast_phase_source", "default_ce")
    else:
        phase_source = job.get("contrast_phase_source", "explicit")

    t0 = time.time()
    timings = {}
    try:
        # ── 1. DICOM → NIfTI if not already present ──────────────────────── #
        nifti_path = CT_DIR / f"{sid}.nii.gz"
        if not nifti_path.exists():
            t_n = time.time()
            from deploy.example_dicom_to_nifti import convert
            convert(job["dicom_dir"], CT_DIR, sid)
            timings["dicom_to_nifti_s"] = time.time() - t_n
        else:
            timings["dicom_to_nifti_s"] = 0.0

        # ── 2. Inference (encoder + probe + router + render) ─────────────── #
        t_i = time.time()
        report = pipeline.generate_report(
            sid,
            skip_llm=True,
            contrast_phase=phase,
            max_findings=int(os.environ.get("CTVLM_MAX_FINDINGS", "12")),
        )
        timings["inference_s"] = time.time() - t_i

        # ── 3. Persist the FullReport JSON for downstream consumers ──────── #
        out_json = REPORTS_DIR / f"{sid}.json"
        out_json.write_text(json.dumps(
            dataclasses.asdict(report), indent=2, default=str
        ))

        _state["last_ok_ts"] = datetime.now(timezone.utc).isoformat()
        _state["jobs_processed"] += 1

        log.info(
            "job_complete  sid=%s  n_positives=%d  latency_s=%.2f  phase=%s/%s",
            sid, len(report.positives), time.time() - t0, phase, phase_source,
        )
        return {
            "status": "ok",
            "study_id": sid,
            "contrast_phase": phase,
            "contrast_phase_source": phase_source,
            "n_positives": len(report.positives),
            "n_findings_total": len(report.probabilities),
            "encoder": report.encoder,
            "probe_path": report.probe_path,
            "summary_text": report.summary_text,
            "report_path": str(out_json),
            "latency_total_s": time.time() - t0,
            **{f"latency_{k}": v for k, v in timings.items()},
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        log.exception("OOM on sid=%s", sid)
        _state["last_err_ts"] = datetime.now(timezone.utc).isoformat()
        return {
            "status": "retry",
            "study_id": sid,
            "error": "cuda_oom",
            "detail": str(e)[:300],
            "stage": "encoder_or_cam",
        }
    except FileNotFoundError as e:
        log.exception("missing input for sid=%s", sid)
        return {
            "status": "reject",
            "study_id": sid,
            "error": "missing_input",
            "detail": str(e)[:300],
        }
    except Exception as e:                                       # noqa: BLE001
        log.exception("error on sid=%s", sid)
        _state["last_err_ts"] = datetime.now(timezone.utc).isoformat()
        return {
            "status": "error",
            "study_id": sid,
            "error": type(e).__name__,
            "detail": str(e)[:500],
            "traceback": traceback.format_exc(limit=10),
        }


# ── health probe ─────────────────────────────────────────────────────────── #

def healthcheck() -> dict:
    return {
        "status": "ready" if _state["model_loaded"] else "loading",
        "model_loaded": _state["model_loaded"],
        "last_ok_ts": _state["last_ok_ts"],
        "last_err_ts": _state["last_err_ts"],
        "jobs_processed": _state["jobs_processed"],
        "gpu_free_gb": torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0.0,
    }


# ── stub broker loop — replace with your actual broker SDK ───────────────── #

def main() -> None:
    """Toy stdin → handle → stdout loop for smoke testing.

    Each line on stdin is a JSON job. Output is one JSON result per line.
    Replace with your broker integration (Redis BLPOP, RabbitMQ consumer,
    SQS poller, NATS subscriber, etc.).
    """
    warmup()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            log.error("bad json: %s", e)
            continue
        result = handle_job(job)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
