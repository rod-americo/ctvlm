"""Typed path configuration loaded from configs/paths.yaml.

Every script imports paths from here; never construct paths inline.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS_YAML = REPO_ROOT / "configs" / "paths.yaml"


@dataclass(frozen=True)
class Paths:
    merlin_root: Path
    work_root: Path
    hf_cache: Path
    rp3d_dir: Path
    kg_dir: Path
    merlin_plus_dir: Path
    masks_dir: Path
    embeddings_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    logs_dir: Path


def _resolve(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


@lru_cache(maxsize=1)
def load_paths(yaml_path: Path | None = None) -> Paths:
    """Load paths, with env-var overrides taking precedence over YAML.

    The deployment worker should set CTVLM_WORK_ROOT and CTVLM_CHECKPOINTS_DIR
    (at minimum) to point at the production data and model locations. See
    docs/02_INSTALLATION.md for the full env list.

    Lookup order per field:
      1. CTVLM_<FIELD> environment variable (e.g. CTVLM_WORK_ROOT)
      2. paths.yaml entry
      3. error if neither is set
    """
    import os

    yaml_path = yaml_path or DEFAULT_PATHS_YAML
    raw: dict = {}
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}

    resolved = {}
    field_names = {f.name for f in fields(Paths)}
    for name in field_names:
        env_key = f"CTVLM_{name.upper()}"
        if env_key in os.environ:
            resolved[name] = _resolve(os.environ[env_key])
        elif name in raw:
            resolved[name] = _resolve(raw[name])
        else:
            raise ValueError(
                f"paths config missing {name!r}; set CTVLM_{name.upper()} env "
                f"var or add it to {yaml_path}"
            )
    return Paths(**resolved)


paths = load_paths()
