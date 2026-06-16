"""Reference Orthanc Python plugin: study-completion → contrast detection
→ broker publish.

Install as Orthanc Python plugin. See https://orthanc.uclouvain.be/book/plugins/python.html
for plugin loading. The relevant entrypoint is `OnChange(changeType, level, resourceId)`.

This file is a self-contained reference. Adapt the publisher block to your
broker SDK (Redis, RabbitMQ, NATS, SQS, ...).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

# Orthanc Python plugin API
try:
    import orthanc  # type: ignore[import-not-found]
except ImportError:                                              # smoke-test outside Orthanc
    orthanc = None  # type: ignore[assignment]

log = logging.getLogger("ctvlm.orthanc_hook")
logging.basicConfig(level=logging.INFO)


# ── configuration ───────────────────────────────────────────────────────── #

STAGING_DIR = Path(os.environ.get("CTVLM_STAGING_DIR", "/staging/ctvlm"))
BROKER_QUEUE = os.environ.get("CTVLM_BROKER_QUEUE", "ctvlm.jobs")

ACCEPTED_BODY_PART_KEYWORDS = ("ABDOMEN", "PELVIS", "ABD")
ACCEPTED_MODALITIES = {"CT"}

# Reject series that are obviously not the primary axial reconstruction
REJECT_IMAGE_TYPE_KEYWORDS = {"DERIVED", "LOCALIZER", "SCREEN_SAVE",
                              "PROJECTION", "DOSE_REPORT"}

# Series-selection priority — prefer venous-phase portal-venous CT, fall back
# to whatever's available. Done case-insensitively against StudyDescription
# and SeriesDescription.
SERIES_PRIORITY_KEYWORDS = (
    ("venous", 10),
    ("portal venous", 10),
    ("portal-venous", 10),
    ("arterial", 5),
    ("delayed", 3),
    ("axial", 1),
)


# ── contrast-phase classifier ───────────────────────────────────────────── #

def classify_contrast_phase(tags: dict) -> tuple[str, str]:
    """Return (phase, source) where phase ∈ {'ce', 'nc', 'unknown'}.

    Mirrors the rule from docs/05_ORTHANC_INTEGRATION.md. The `source` field
    records why we made the choice so an operator can audit.
    """
    agent = (tags.get("ContrastBolusAgent") or "").strip()
    try:
        volume = float(tags.get("ContrastBolusVolume") or 0)
    except (TypeError, ValueError):
        volume = 0.0
    route = (tags.get("ContrastBolusRoute") or "").strip().upper()

    if agent and volume > 0 and route in {"", "IV", "INTRAVENOUS"}:
        return "ce", "ContrastBolusAgent+Volume"

    if not agent and volume == 0:
        desc = " ".join(filter(None, [
            tags.get("StudyDescription") or "",
            tags.get("SeriesDescription") or "",
        ])).lower()
        if re.search(r"non[-\s]?contrast|without iv|without contrast", desc):
            return "nc", "description_keyword"
        return "unknown", "no_iv_contrast_no_keyword"

    return "unknown", "ambiguous"


# ── series selection ────────────────────────────────────────────────────── #

def _passes_filters(series_tags: dict) -> bool:
    modality = (series_tags.get("Modality") or "").upper()
    if modality not in ACCEPTED_MODALITIES:
        return False
    body = (series_tags.get("BodyPartExamined") or "").upper()
    if not any(kw in body for kw in ACCEPTED_BODY_PART_KEYWORDS):
        return False
    image_type = (series_tags.get("ImageType") or "").upper()
    if any(kw in image_type for kw in REJECT_IMAGE_TYPE_KEYWORDS):
        return False
    try:
        slice_mm = float(series_tags.get("SliceThickness") or 0)
        if slice_mm <= 0 or slice_mm > 5.5:
            return False
    except (TypeError, ValueError):
        return False
    return True


def _series_score(series_tags: dict) -> int:
    desc = " ".join(filter(None, [
        series_tags.get("SeriesDescription") or "",
        series_tags.get("StudyDescription") or "",
    ])).lower()
    score = 0
    for kw, w in SERIES_PRIORITY_KEYWORDS:
        if kw in desc:
            score = max(score, w)
    # Tie-break by latest instance number (newest acquisition)
    try:
        score = score * 10000 + int(series_tags.get("SeriesNumber") or 0)
    except (TypeError, ValueError):
        pass
    return score


# ── DICOM staging ───────────────────────────────────────────────────────── #

def stage_series_to_disk(study_id: str, series_id: str) -> Path:
    """Pull every instance in the chosen series into a flat directory the
    worker can read. Orthanc's own storage may not be reachable from the
    worker filesystem; we materialise the series under STAGING_DIR."""
    out_dir = STAGING_DIR / study_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = json.loads(orthanc.RestApiGet(  # type: ignore[union-attr]
        f"/series/{series_id}/instances"))
    for inst in instance_ids:
        inst_id = inst["ID"]
        dcm_bytes = orthanc.RestApiGet(f"/instances/{inst_id}/file")  # type: ignore[union-attr]
        (out_dir / f"{inst_id}.dcm").write_bytes(dcm_bytes)
    log.info("staged %d instances → %s", len(instance_ids), out_dir)
    return out_dir


# ── broker publish (REPLACE WITH YOUR BROKER) ───────────────────────────── #

def publish_to_broker(payload: dict) -> None:
    """Replace with your actual broker. This stub just logs the JSON."""
    log.info("broker_publish queue=%s payload=%s",
             BROKER_QUEUE, json.dumps(payload))
    # ── example: redis ──────────────────────────────────────────────────── #
    # import redis
    # r = redis.Redis.from_url(os.environ["CTVLM_REDIS_URL"])
    # r.lpush(BROKER_QUEUE, json.dumps(payload))
    # ── example: rabbitmq ──────────────────────────────────────────────── #
    # import pika
    # conn = pika.BlockingConnection(...)
    # ch = conn.channel()
    # ch.basic_publish(exchange='', routing_key=BROKER_QUEUE,
    #                   body=json.dumps(payload),
    #                   properties=pika.BasicProperties(delivery_mode=2))


# ── main hook ───────────────────────────────────────────────────────────── #

def OnChange(changeType, level, resourceId):                     # noqa: N802 — Orthanc API
    """Orthanc callback. We only care about OnStableStudy events."""
    if changeType != orthanc.ChangeType.STABLE_STUDY:            # type: ignore[union-attr]
        return

    study_tags = json.loads(orthanc.RestApiGet(                  # type: ignore[union-attr]
        f"/studies/{resourceId}/shared-tags"))
    study_main = json.loads(orthanc.RestApiGet(                  # type: ignore[union-attr]
        f"/studies/{resourceId}"))
    accession = study_tags.get("AccessionNumber") or \
                study_main.get("MainDicomTags", {}).get("StudyInstanceUID")
    if not accession:
        log.warning("study %s: no accession", resourceId)
        return

    # Enumerate series, filter, score
    series_list = []
    for series_id in study_main["Series"]:
        s_tags = json.loads(orthanc.RestApiGet(                  # type: ignore[union-attr]
            f"/series/{series_id}/shared-tags"))
        if _passes_filters(s_tags):
            series_list.append((series_id, s_tags, _series_score(s_tags)))

    if not series_list:
        log.info("study %s: no accepted series, skipping", accession)
        return

    # Pick the highest-scored series
    series_list.sort(key=lambda r: -r[2])
    chosen_series_id, chosen_tags, chosen_score = series_list[0]
    phase, phase_source = classify_contrast_phase(chosen_tags)

    # Materialise to disk and publish
    staged_dir = stage_series_to_disk(accession, chosen_series_id)
    publish_to_broker({
        "study_id": accession,
        "orthanc_study_uuid": resourceId,
        "orthanc_series_uuid": chosen_series_id,
        "modality": chosen_tags.get("Modality"),
        "body_part": chosen_tags.get("BodyPartExamined"),
        "slice_thickness_mm": chosen_tags.get("SliceThickness"),
        "contrast_phase": phase,
        "contrast_phase_source": phase_source,
        "dicom_dir": str(staged_dir),
        "series_score": chosen_score,
    })


# Register with Orthanc
if orthanc is not None:
    orthanc.RegisterOnChangeCallback(OnChange)
    log.info("ctvlm Orthanc hook registered")
