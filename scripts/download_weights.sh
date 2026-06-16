#!/usr/bin/env bash
# Download Merlin + Pillar-0 encoder weights from HuggingFace into the local HF cache.
#
# Prerequisites:
#   - huggingface_hub installed (comes with the `transformers` install)
#   - HF_TOKEN env var set to a token whose account has accepted:
#       https://huggingface.co/stanfordmimi/Merlin       (MIT)
#       https://huggingface.co/YalaLab/Pillar0-AbdomenCT (permissive)
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./scripts/download_weights.sh
#   ./scripts/download_weights.sh --hf-cache /custom/path
#
# After this script: $HF_HOME (default ./hf_cache) contains both encoder weights
# and the worker can boot fully offline (HF_HUB_OFFLINE=1).

set -euo pipefail

HF_CACHE="${HF_HOME:-$(pwd)/hf_cache}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-cache) HF_CACHE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN env var not set." >&2
    echo "  1. Sign in at https://huggingface.co" >&2
    echo "  2. Accept the license at https://huggingface.co/stanfordmimi/Merlin" >&2
    echo "  3. Accept the license at https://huggingface.co/YalaLab/Pillar0-AbdomenCT" >&2
    echo "  4. Create a token at https://huggingface.co/settings/tokens" >&2
    echo "  5. export HF_TOKEN=hf_..." >&2
    exit 1
fi

mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

echo "==> Downloading Merlin (stanfordmimi/Merlin) → $HF_CACHE"
python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='stanfordmimi/Merlin',
    cache_dir=os.environ['HF_HOME'],
    token=os.environ['HF_TOKEN'],
)
print('  Merlin OK')
"

echo "==> Downloading Pillar-0 (YalaLab/Pillar0-AbdomenCT) → $HF_CACHE"
python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='YalaLab/Pillar0-AbdomenCT',
    cache_dir=os.environ['HF_HOME'],
    token=os.environ['HF_TOKEN'],
)
print('  Pillar-0 OK')
"

# Merlin's pip package additionally caches a checkpoint inside its own dir on first
# load. We can let that happen at runtime, or pre-trigger it offline:
echo "==> Pre-loading Merlin via its Python package (first-load checkpoint pull)"
python -c "
import os
os.environ.setdefault('HF_HOME', '${HF_CACHE}')
from merlin import Merlin
m = Merlin(ImageEmbedding=True)
print('  Merlin pkg checkpoint cached')
" 2>/dev/null || echo "  (skipped — install ctvlm + merlin package first; run again after pip install -e .)"

echo
echo "Done. Weights are in: $HF_CACHE"
echo
echo "Next: verify offline load works"
echo "  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test.py"
