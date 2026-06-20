#!/usr/bin/env bash
# Container startup script — runs before the viewer / worker process.
#
# Idempotently:
#   1. Verifies the concat probe checkpoint is present.
#   2. Downloads Merlin + Pillar-0 encoder weights from HuggingFace if not cached.
#   3. Execs whatever command was passed (`exec "$@"`).
#
# This is the ENTRYPOINT for both viewer and worker images. Persisting weights
# across container restarts is handled by mounting /opt/ctvlm/hf_cache as a
# volume (see docker-compose.yml).
#
# On first launch with no cache: pulls ~1.5 GB from HF (Merlin 1.1 GB + Pillar-0
# ~356 MB). Subsequent launches are instant.

set -euo pipefail

CTVLM_ROOT="/opt/ctvlm"
HF_HOME="${HF_HOME:-$CTVLM_ROOT/hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
CKPT_DIR="${CTVLM_CHECKPOINTS_DIR:-$CTVLM_ROOT/checkpoints}"
export HF_HOME HF_HUB_CACHE

# ── 1. Probe checkpoint sanity ──────────────────────────────────────────── #
if [ ! -f "$CKPT_DIR/concat_rate_probe.pt" ]; then
    echo "ERROR: concat probe checkpoint missing at $CKPT_DIR/concat_rate_probe.pt" >&2
    echo "  Bind-mount ./checkpoints/ from the repo into $CKPT_DIR, or" >&2
    echo "  copy concat_rate_probe.pt into $CKPT_DIR before running." >&2
    exit 1
fi

# ── 2. Encoder weights ──────────────────────────────────────────────────── #
PILLAR0_REPO_DIR="$HF_HUB_CACHE/models--YalaLab--Pillar0-AbdomenCT"

# Merlin's pip package stores its .pt inside the package itself; the Docker
# image symlinks that package directory to /opt/ctvlm/hf_cache/merlin_pkg/, so
# caching is shared with the mounted HF cache volume.
MERLIN_PKG_CKPT_DIR="$CTVLM_ROOT/hf_cache/merlin_pkg"

need_download=0
if [ ! -d "$PILLAR0_REPO_DIR" ] || ! ls "$PILLAR0_REPO_DIR/snapshots"/*/*.safetensors* >/dev/null 2>&1; then
    need_download=1
    echo "  [startup] Pillar-0 weights not cached"
fi
if [ ! -d "$MERLIN_PKG_CKPT_DIR" ] || ! ls "$MERLIN_PKG_CKPT_DIR"/*.pt >/dev/null 2>&1; then
    need_download=1
    echo "  [startup] Merlin weights not cached"
fi

if [ "$need_download" -eq 1 ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "" >&2
        echo "ERROR: encoder weights missing AND HF_TOKEN not set." >&2
        echo "" >&2
        echo "  1. Sign in at https://huggingface.co" >&2
        echo "  2. Accept https://huggingface.co/stanfordmimi/Merlin       (MIT)" >&2
        echo "  3. Accept https://huggingface.co/YalaLab/Pillar0-AbdomenCT (permissive)" >&2
        echo "  4. Create a token at https://huggingface.co/settings/tokens" >&2
        echo "  5. Pass it through docker-compose .env file:" >&2
        echo "       echo 'HF_TOKEN=hf_...' >> .env" >&2
        echo "" >&2
        exit 1
    fi
    echo "  [startup] downloading encoder weights (one-time, ~1.5 GB) ..."
    "$CTVLM_ROOT/scripts/download_weights.sh" --hf-cache "$HF_HOME"
    echo "  [startup] download complete"
else
    echo "  [startup] encoder weights already cached"
fi

# ── 3. Exec the actual command ──────────────────────────────────────────── #
echo "  [startup] launching: $*"
exec "$@"
