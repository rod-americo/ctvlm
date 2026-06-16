"""Pillar-0 (YalaLab Atlas) abdomen-CT encoder + per-organ token pooling.

Validated as the Phase 2 structure encoder — see `reports/pillar0_probe.md` and
[[pillar0]] in memory. A single forward yields the model's global embedding *and*,
via a hook, the **pre-pool token grid**; pooling those tokens inside MerlinPlus organ
masks gives **per-organ node features** for the Phase 3 GNN — no per-organ forward
passes. The pool operator matters (linear-probe study): focal findings favour MAX,
diffuse/size favour MEAN, so node features concatenate [mean, max] per organ.

External assets (not in the repo; ECL-2.0 / permissive). Override via env if needed:
  PILLAR0_REPO  HF repo id of the gated weights (needs an accepted-gate token)
  RAVE_WIN      path to rad-vision-engine's windowing_utils.py (the 11 CT windows)

Pipeline (details in memory/pillar0.md): canonical-RAS CT -> 384^3 (GPU resize) ->
rave 11 CT windows -> Atlas tower. Load by HF repo id with trust_remote_code (NOT a
local path); embed via extract_vision_feats(image={"abdomen_ct": x}); autocast fp16
(do NOT .half() the model). Measured: ~8.6 GB VRAM, <0.5 s/forward on a 3090.

Public API:
    load_model() -> torch.nn.Module                      # cached singleton, eval/cuda
    preprocess(study_id, *, ct_img=None) -> Tensor       # (1, 11, 384, 384, 384) on GPU
    embed(study_id|x, model) -> (global_max, global_mean, token_grid)
    organ_features(study_id, model, organs, ops=("mean","max")) -> {organ: {op: vec}}
"""
from __future__ import annotations

import os
import importlib.util
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from src.data import merlinplus as mp

PILLAR0_REPO = os.environ.get("PILLAR0_REPO", "YalaLab/Pillar0-AbdomenCT")
RAVE_WIN = os.environ.get(
    "RAVE_WIN", "/mnt/e/ctvlm/rave/vision_engine/utils/windowing_utils.py")
SIZE = 384                       # model image_size (isotropic)
MODALITY = "abdomen_ct"
EMBED_DIM = 1152                 # concat of 3 scales x 384
# rave get_available_windows("CT") order — 10 anatomical + minmax = 11 input channels.
CT_WINDOWS = ["lung", "mediastinum", "abdomen", "liver", "bone", "brain",
              "subdural", "stroke", "temporal_bone", "soft_tissue", "minmax"]


@lru_cache(maxsize=1)
def _windowing():
    """rave's apply_windowing, loaded standalone (its package __init__ needs lz4/SITK)."""
    spec = importlib.util.spec_from_file_location("rave_win", RAVE_WIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_windowing


def _patch_tx5_tied_weights_compat() -> None:
    """transformers 5.x renamed the tied-weights API: `_finalize_model_loading` now
    expects `self.all_tied_weights_keys` to be a dict-like with `.keys()`. Pillar-0's
    custom `CLIPMultimodalAtlas` class predates that rename and doesn't define it,
    so loading crashes with `AttributeError: no attribute 'all_tied_weights_keys'`.

    Empty-dict is the correct default — Pillar-0's vision/text towers are independent,
    no tied weights to preserve. Idempotent (guarded by a sentinel attr).
    """
    import transformers.modeling_utils as _mu
    if getattr(_mu.PreTrainedModel, "_pillar0_compat_patched", False):
        return
    _orig = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
    def _patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, *args, **kwargs)
    _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched
    _mu.PreTrainedModel._pillar0_compat_patched = True


@lru_cache(maxsize=1)
def load_model():
    """Load Pillar-0 once (eval, cuda, fp32 — autocast handles fp16 at call time).

    Loads by HF repo id with trust_remote_code; a local path trips the dynamic-module
    relative-import resolver. Needs a token whose account accepted the model gate.
    """
    _patch_tx5_tied_weights_compat()
    from transformers import AutoModel
    token = None
    tok_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(tok_path):
        token = open(tok_path).read().strip()
    model = AutoModel.from_pretrained(PILLAR0_REPO, trust_remote_code=True, token=token)
    return model.eval().cuda()


def preprocess(study_id: str, *, ct_img=None) -> torch.Tensor | None:
    """Canonical-RAS CT -> 384^3 (GPU) -> 11 CT windows -> (1, 11, 384, 384, 384) on GPU.

    Canonical RAS so the volume shares a grid with MerlinPlus masks (organ pooling).
    """
    if ct_img is None:
        ct_img = mp.load_ct(study_id)
        if ct_img is None:
            return None
    vol = np.asanyarray(ct_img.dataobj).astype(np.float32)          # RAS (X,Y,Z) HU
    vol = np.ascontiguousarray(np.transpose(vol, (2, 1, 0)))        # (D,H,W)
    t = torch.from_numpy(vol)[None, None].cuda()
    t = F.interpolate(t, size=(SIZE,) * 3, mode="trilinear", align_corners=False)[0, 0]
    return _windowing()(t, "all", "CT").unsqueeze(0)                # (1, 11, 384^3)


def _capture_tokens(model, x):
    """Run one forward, returning (global_embedding, [per-scale token tensors (C,N)])."""
    grabbed = []
    h = model.model.visual.maxpool.register_forward_pre_hook(
        lambda _m, inp: grabbed.append(inp[0].detach()))
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            emb = model.extract_vision_feats(image={MODALITY: x})
    finally:
        h.remove()
    return emb, [g.float().squeeze(0) for g in grabbed]


def embed(study_id_or_x, model):
    """Return (global_max, global_mean, token_grid) for a study or a preprocessed tensor.

    global_* are 1152-d numpy; token_grid is the finest scale as (g, g, g, C) numpy
    (g~64), aligned with canonical-RAS masks for organ pooling.
    """
    x = study_id_or_x if torch.is_tensor(study_id_or_x) else preprocess(study_id_or_x)
    if x is None:
        return None
    _, scales = _capture_tokens(model, x)
    gmax = torch.cat([s.max(1).values for s in scales]).cpu().numpy()
    gmean = torch.cat([s.mean(1) for s in scales]).cpu().numpy()
    fine = scales[0]
    C, n0 = fine.shape
    g = round(n0 ** (1 / 3))
    grid = fine.transpose(0, 1).reshape(g, g, g, C).cpu().numpy()
    return gmax, gmean, grid


def _topk_mean(toks: np.ndarray, k_frac: float = 0.10) -> np.ndarray:
    """Per-dimension mean of the top-k tokens by that dim's activation. Catches focal
    signal (peak activity in a small token cluster) without single-voxel max noise.
    k_frac=0.10 = top 10% of tokens per dim; min k=1."""
    n = toks.shape[0]
    k = max(1, int(round(n * k_frac)))
    if k >= n:
        return toks.mean(0)
    # np.partition is O(N) per dim and gives the top-k unsorted; mean is order-invariant.
    return np.partition(toks, n - k, axis=0)[-k:].mean(0)


def _organ_grid(study_id, organ, g):
    """MerlinPlus organ mask -> token-grid resolution (g^3) bool, same transform as CT."""
    try:
        m = mp.load_mask(study_id, organ)
    except FileNotFoundError:
        return None
    m = np.ascontiguousarray(np.transpose(m, (2, 1, 0)).astype(np.float32))
    md = F.interpolate(torch.from_numpy(m)[None, None].cuda(), size=(g,) * 3,
                       mode="nearest")[0, 0]
    return (md > 0.5).cpu().numpy()


def _pool_organs(study_id, grid, organs, ops) -> dict:
    g = grid.shape[0]
    feats: dict[str, dict] = {}
    for organ in organs:
        mk = _organ_grid(study_id, organ, g)
        if mk is None or not mk.any():
            continue
        toks = grid[mk]                       # (k, C)
        d = {}
        if "mean" in ops:
            d["mean"] = toks.mean(0)
        if "max" in ops:
            d["max"] = toks.max(0)
        if "topk" in ops:
            d["topk"] = _topk_mean(toks)
        feats[organ] = d
    return feats


def scale_tokens(study_id_or_x, model, scales=(1,)) -> dict | None:
    """Pre-pool tokens per Atlas scale as {scale: (N, C) float16} — for a Q-Former.

    Scales (measured at 384^3): 0 -> 64^3=262144, 1 -> 16^3=4096, 2 -> 4^3=64 tokens,
    each C=384. Scale-1 is the usual spatial-but-tractable resampler key/value source.
    """
    import numpy as _np
    x = study_id_or_x if torch.is_tensor(study_id_or_x) else preprocess(study_id_or_x)
    if x is None:
        return None
    _, captured = _capture_tokens(model, x)                      # list of (C, N)
    return {s: captured[s].transpose(0, 1).to(torch.float16).cpu().numpy()  # (N, C)
            for s in scales if s < len(captured)}


def pool_grid(token_grid, organ_grids: dict, ops=("mean", "max")) -> dict:
    """Pool a token grid with PRECOMPUTED organ masks {organ: bool grid at token res}.

    For the parallel pipeline where CPU workers downsample the masks. Returns
    {organ: {op: vec}} for non-empty masks.
    """
    feats: dict[str, dict] = {}
    for organ, mk in organ_grids.items():
        if mk is None or not mk.any():
            continue
        toks = token_grid[mk]
        d = {}
        if "mean" in ops:
            d["mean"] = toks.mean(0)
        if "max" in ops:
            d["max"] = toks.max(0)
        if "topk" in ops:
            d["topk"] = _topk_mean(toks)
        feats[organ] = d
    return feats


def extract(study_id, model, organs=(), ops=("mean", "max")) -> dict | None:
    """One forward → everything: {'gmax', 'gmean', 'organ': {organ: {op: vec}}}.

    `gmax`/`gmean` are 1152-d global pools; `organ` holds per-organ [mean/max] token
    pools (the GNN node features). This is the primary Phase 2 entry point.
    """
    out = embed(study_id, model)
    if out is None:
        return None
    gmax, gmean, grid = out
    return {"gmax": gmax, "gmean": gmean,
            "organ": _pool_organs(study_id, grid, organs, ops)}


def organ_features(study_id, model, organs, ops=("mean", "max")) -> dict:
    """Per-organ node features {organ: {op: vector}} — concat the ops for the GNN."""
    out = extract(study_id, model, organs, ops)
    return out["organ"] if out else {}
