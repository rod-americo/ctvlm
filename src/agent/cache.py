"""Tiny on-disk cache for tool calls.

Each tool result (JSON-serialisable) gets one file under
`/mnt/e/ctvlm/agent_cache/<sid>/<tool>__<args_hash>.json`. Heatmap NIfTIs and
mask arrays keep their existing NIfTI/.npz caches in `/mnt/e/ctvlm/heatmaps/`
and MerlinPlus; only path + summary stats land in the JSON here.

Re-running the pipeline for the same case touches zero GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.config import paths

CACHE_ROOT = paths.work_root / "agent_cache"


def _args_hash(args: dict[str, Any]) -> str:
    """Stable hash of the args dict (sorted keys, json-serialisable)."""
    blob = json.dumps(args, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def cache_path(sid: str, tool: str, args: dict[str, Any]) -> Path:
    h = _args_hash(args)
    return CACHE_ROOT / sid / f"{tool}__{h}.json"


def cached(sid: str, tool: str, args: dict[str, Any], fn: Callable[[], Any]) -> tuple[Any, bool]:
    """Returns (result, cache_hit). Stores fn() output on cache miss.

    fn must return a JSON-serialisable value (or one this function can fall back to
    str() for — e.g. small dicts of floats/ints/strings/lists are fine).
    """
    p = cache_path(sid, tool, args)
    if p.exists():
        try:
            return json.loads(p.read_text()), True
        except json.JSONDecodeError:
            pass
    result = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(result, default=str))
    except (TypeError, ValueError):
        # last-resort: stringify
        p.write_text(json.dumps({"_repr": str(result)}))
    return result, False
