# syntax=docker/dockerfile:1.6
#
# ctvlm — agentic radiology report generator for CT abdomen-pelvis
#
# Encoder weights (Merlin + Pillar-0) are downloaded from HuggingFace on FIRST
# container start (one-time, ~1.5 GB) via scripts/startup.sh — provided HF_TOKEN
# is set and the account has accepted both gates:
#     https://huggingface.co/stanfordmimi/Merlin       (MIT)
#     https://huggingface.co/YalaLab/Pillar0-AbdomenCT (permissive)
#
# Persist the download across restarts by mounting hf_cache as a volume
# (handled by docker-compose.yml).
#
# Build:
#   docker build -t ctvlm:0.2.0 .
#   docker build -t ctvlm-viewer:0.2.0 --target viewer .
#
# Run viewer (auto-downloads on first start):
#   docker compose up viewer
#   # → http://localhost:8501

ARG CUDA_TAG=12.4.1-cudnn-devel-ubuntu22.04

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — base: CUDA + Python + ctvlm production code + symlinked cache
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_TAG} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/opt/ctvlm/hf_cache \
    CTVLM_WORK_ROOT=/opt/ctvlm/work \
    CTVLM_CHECKPOINTS_DIR=/opt/ctvlm/checkpoints \
    CTVLM_HF_CACHE=/opt/ctvlm/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates \
        dcm2niix \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --upgrade pip wheel setuptools

WORKDIR /opt/ctvlm

# Install ctvlm. Two-step copy lets Docker cache the dep install across code-only changes.
COPY pyproject.toml ./
RUN python -m pip install --no-deps -e . || true

# Copy code + data + configs + scripts
COPY src/         ./src/
COPY data/        ./data/
COPY configs/     ./configs/
COPY scripts/     ./scripts/
COPY tests/       ./tests/
COPY checkpoints/ ./checkpoints/
COPY README.md ./
COPY docs/ ./docs/

# Install deps now that source is present
RUN python -m pip install -e .

# Symlink Merlin's pip-package internal checkpoint dir into HF_HOME, so the
# 1.1 GB Merlin .pt persists in the same volume as the Pillar-0 HF cache.
# (The merlin pkg writes its checkpoint to <pkg>/models/checkpoints/ on first
# load — we redirect that to a volume-friendly path.)
RUN MERLIN_PKG=$(python -c "import merlin, pathlib; print(pathlib.Path(merlin.__file__).parent)" 2>/dev/null) \
    && mkdir -p /opt/ctvlm/hf_cache/merlin_pkg \
    && if [ -n "$MERLIN_PKG" ] && [ -d "$MERLIN_PKG/models" ]; then \
           rm -rf "$MERLIN_PKG/models/checkpoints" \
        && ln -s /opt/ctvlm/hf_cache/merlin_pkg "$MERLIN_PKG/models/checkpoints" \
        && echo "Symlinked Merlin pkg checkpoints → /opt/ctvlm/hf_cache/merlin_pkg"; \
       else \
           echo "WARNING: merlin pip pkg not found, skipping symlink (download_weights.sh will pull at startup)"; \
       fi || true

# Working directories the worker / viewer expect
RUN mkdir -p /opt/ctvlm/work/{ct_volumes,organ_masks,merlin_global,pillar0_emb,heatmaps,agent_cache,reports_out} \
             /opt/ctvlm/hf_cache

# Make startup script executable
RUN chmod +x /opt/ctvlm/scripts/startup.sh /opt/ctvlm/scripts/download_weights.sh

# Image-build smoke (no GPU, no network — just imports)
RUN python -c "from src.agent import pipeline, recipes, render; from src.agent.canonical import CANONICAL_TO_CATEGORY; print('ctvlm imports OK; canonicals=', len(CANONICAL_TO_CATEGORY))"

# Default ENTRYPOINT handles the weight download on first run
ENTRYPOINT ["/opt/ctvlm/scripts/startup.sh"]
CMD ["python", "-c", "print('ctvlm:0.2.0 — see README.md and docs/ for entry points')"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — viewer: base + streamlit + sidecar file server
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS viewer

# Streamlit + plotting extras
RUN python -m pip install "streamlit>=1.30" "plotly>=5.18"

# Viewer assets
COPY viewer/ ./viewer/

EXPOSE 8501 8502

# Viewer-specific entry: boots file server in background, then the user-facing
# Streamlit web_app in foreground. The startup.sh ENTRYPOINT runs first →
# downloads encoder weights on first launch → then execs this CMD.
#
# To run the developer-oriented dashboard instead of web_app, override CMD:
#   docker compose run --rm viewer streamlit run viewer/streamlit_app.py
COPY <<'EOF' /opt/ctvlm/viewer/run.sh
#!/usr/bin/env bash
set -euo pipefail
cd /opt/ctvlm

# Sidecar file server for NIfTI / heatmap streaming + NiiVue viewer page
python viewer/ct_files_server.py --port 8502 &
CTFS_PID=$!

# User-facing Streamlit app (foreground)
exec streamlit run viewer/web_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 2000
EOF
RUN chmod +x /opt/ctvlm/viewer/run.sh

ENV PYTHONPATH=/opt/ctvlm

CMD ["/opt/ctvlm/viewer/run.sh"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — worker: base + minimal extras for GPU broker integration
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS worker

RUN python -m pip install "prometheus-client>=0.20" || true

COPY deploy/ ./deploy/

CMD ["python", "deploy/example_worker.py"]
