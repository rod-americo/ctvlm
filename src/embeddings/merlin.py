"""Stanford Merlin (Blankemeier et al.) abdominal CT encoder — head-to-head baseline.

Released checkpoint: `stanfordmimi/Merlin` (MIT). Image tower is an I3D-inflated
ResNet-152 trained on Stanford's Merlin dataset (15K abdominal CTs paired with reports
+ EHR). We use it as a *baseline against* our Pillar-0 + organ-pool + R-GAT stack:

  - **Global 2048-d embedding** — Merlin's stock image embedding, the input for a
    linear/MLP probe over our 50-finding taxonomy.
  - **Layer3 pre-pool feature map** `(1024, 20, 14, 14)` — hooked at
    `model.encode_image.i3_resnet.layer3`. Used as the per-token feature for the
    drop-in "Merlin tokens -> organ pool -> R-GAT" experiment, the same pipeline as
    `scripts/18` but with Merlin tokens instead of Pillar-0 tokens.

Caveats:
  - **Spatial resolution is coarse.** layer3 = 20x14x14 = 3,920 tokens; layer4 = 490.
    For comparison Pillar-0 keeps a 64^3 = 262,144-token map at the finest scale.
    Small organs (gall bladder, adrenals, esophagus) often get <5 tokens after the
    mask is downsampled to (20,14,14) — a real limit of this encoder for organ pools.
  - **Preprocessing differs from Pillar-0.** Merlin expects RAS, 1.5/1.5/3.0 mm
    spacing, HU clipped to [-1000, 1000] and rescaled to [0,1], pad+center-crop to
    224x224x160. We replicate that here (MONAI used implicitly in `Merlin.dataloader`).
  - **License:** MIT for code/weights; fine for downstream use.

Public API:
    load_model() -> torch.nn.Module                       # cached singleton, eval/cuda
    preprocess(study_id) -> Tensor                        # (1, 1, 224, 224, 160) on GPU
    embed(study_id|x, model) -> (global_2048, layer3_grid) # fp32 numpy
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from src.data import merlinplus as mp
from src.config import paths

# Merlin's MONAI preprocessing constants (from merlin/data/monai_transforms.py).
TARGET_SPACING = (1.5, 1.5, 3.0)     # mm (RAS axes order: R, A, S)
TARGET_SHAPE = (224, 224, 160)       # (R, A, S) after pad + center-crop
HU_RANGE = (-1000.0, 1000.0)         # window before [0,1] rescale

# Hook target for the pre-pool feature map (3920 tokens at 1024-d).
LAYER_PATH = "model.encode_image.i3_resnet.layer3"


def _get_submodule(m: torch.nn.Module, dotted: str) -> torch.nn.Module:
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


@lru_cache(maxsize=1)
def load_model():
    """Load Merlin's image-only encoder once (eval, cuda, fp32 weights — autocast at call)."""
    os.environ.setdefault("HF_HOME", str(paths.hf_cache))
    from merlin import Merlin
    model = Merlin(ImageEmbedding=True).eval().cuda()
    return model


def _resample_to_spacing(vol_RAS_f32: torch.Tensor, src_spacing_mm) -> torch.Tensor:
    """Resample (R, A, S) volume to TARGET_SPACING mm via trilinear interpolation.

    `vol` is (R, A, S) on CUDA float32. `src_spacing_mm` is the (R, A, S) voxel size
    of the input. Output is (R', A', S') at TARGET_SPACING.
    """
    factors = tuple(s / t for s, t in zip(src_spacing_mm, TARGET_SPACING))
    new_shape = tuple(max(1, int(round(d * f))) for d, f in zip(vol_RAS_f32.shape, factors))
    t = vol_RAS_f32[None, None]  # (1, 1, R, A, S)
    t = F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    return t[0, 0]


def _pad_center_crop(vol: torch.Tensor, target=TARGET_SHAPE) -> torch.Tensor:
    """Pad with 0 (post-scale: ~[-1000 HU]) then center-crop to `target`. Matches MONAI's
    SpatialPadd + CenterSpatialCropd. Input/output are (R, A, S) tensors.
    """
    out = vol
    # Pad each dim that's too small to at least `target`.
    pads = []  # last-dim-first for F.pad
    for d_in, d_t in zip(reversed(out.shape), reversed(target)):
        diff = max(0, d_t - d_in)
        pads += [diff // 2, diff - diff // 2]
    if any(pads):
        out = F.pad(out, pads, mode="constant", value=0.0)
    # Center-crop each dim that's too large.
    slices = []
    for d_in, d_t in zip(out.shape, target):
        start = max(0, (d_in - d_t) // 2)
        slices.append(slice(start, start + d_t))
    return out[tuple(slices)]


def preprocess(study_id: str, *, ct_img=None) -> torch.Tensor | None:
    """Merlin's input pipeline: canonical RAS -> 1.5/1.5/3mm -> HU clip+scale ->
    pad+center-crop to 224x224x160 -> (1, 1, R, A, S) on GPU as float32.
    """
    if ct_img is None:
        ct_img = mp.load_ct(study_id)
        if ct_img is None:
            return None
    # canonical RAS is guaranteed by mp.load_ct (-> as_closest_canonical).
    vol = np.asanyarray(ct_img.dataobj).astype(np.float32)        # (R, A, S)
    # Voxel spacing from the affine — the diagonal of the linear part.
    aff = ct_img.affine
    spacing = tuple(float(abs(aff[i, i])) for i in range(3))      # (R, A, S) mm

    t = torch.from_numpy(vol).cuda()
    t = _resample_to_spacing(t, spacing)
    t = t.clamp_(HU_RANGE[0], HU_RANGE[1])
    t = (t - HU_RANGE[0]) / (HU_RANGE[1] - HU_RANGE[0])           # [0, 1]
    t = _pad_center_crop(t, TARGET_SHAPE)
    return t[None, None]                                           # (1, 1, R, A, S)


def _capture(model, x):
    """One forward; return (global_2048, layer3_grid as fp32 numpy)."""
    feats: dict = {}

    def hook(_m, _inp, out):
        feats["layer3"] = out.detach()

    h = _get_submodule(model, LAYER_PATH).register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x)
    finally:
        h.remove()
    # out: (1, 1, 2048); feats["layer3"]: (1, 1024, D', H', W')
    g = out.float().squeeze().cpu().numpy()                       # (2048,)
    grid = feats["layer3"].float().squeeze(0).cpu().numpy()       # (1024, D', H', W')
    return g, grid


def embed(study_id_or_x, model):
    """Return (global_2048, layer3_grid) for one study or a preprocessed tensor.

    `global_2048` is the model's stock image embedding (fp32 1-D). `layer3_grid` is the
    pre-`avgpool` feature map (1024, ~20, ~14, ~14) — the input for the organ pool.
    """
    x = study_id_or_x if torch.is_tensor(study_id_or_x) else preprocess(study_id_or_x)
    if x is None:
        return None
    return _capture(model, x)


def organ_pool(layer3_grid: np.ndarray, study_id: str, organs, ops=("mean", "max")) -> dict:
    """Pool layer3 features inside MerlinPlus organ masks (downsampled to grid size).

    `layer3_grid` is (C, D, H, W) numpy. We rebuild an (R, A, S) view via Merlin's own
    axis ordering of the I3D ResNet input — see preprocess for the chain. The grid is
    aligned with the *post-preprocess* CT volume (post pad+center-crop, post resample),
    not the original CT, so we must replicate the same transform on each organ mask.
    """
    # Layer3 grid is (C, D=20, H=14, W=14) where the input was (R, A, S) = (224, 224, 160).
    # I3D-inflated ResNet-152: the H,W axes downsample 16x; the depth axis downsamples
    # 8x (Merlin's I3D inflation pattern). So D corresponds to S/8, H to R/16, W to A/16.
    # The grid axes therefore map: D <-> S, H <-> R, W <-> A.
    C, gD, gH, gW = layer3_grid.shape
    # Build organ masks at the same spatial layout the grid lives in (D, H, W) = (S, R, A).
    feats: dict[str, dict] = {}
    ct_img = mp.load_ct(study_id)
    if ct_img is None:
        return feats
    aff = ct_img.affine
    spacing = tuple(float(abs(aff[i, i])) for i in range(3))      # (R, A, S) mm
    for organ in organs:
        try:
            m = mp.load_mask(study_id, organ)                     # (R, A, S) bool
        except FileNotFoundError:
            continue
        if not m.any():
            continue
        t = torch.from_numpy(m.astype(np.float32)).cuda()
        t = _resample_to_spacing(t, spacing)                      # to TARGET_SPACING grid
        t = _pad_center_crop(t, TARGET_SHAPE)                     # (R, A, S) = (224,224,160)
        # Reorder to (S, R, A) so dims match the grid (D, H, W).
        t = t.permute(2, 0, 1).contiguous()                       # (S, R, A)
        # Downsample mask to grid size with nearest-neighbour.
        t = F.interpolate(t[None, None], size=(gD, gH, gW), mode="nearest")[0, 0]
        mk = (t > 0.5).cpu().numpy()
        if not mk.any():
            continue
        # (gD, gH, gW) -> bool; gather tokens as (n, C)
        toks = layer3_grid.transpose(1, 2, 3, 0)[mk]              # (n, C)
        d = {}
        if "mean" in ops:
            d["mean"] = toks.mean(0)
        if "max" in ops:
            d["max"] = toks.max(0)
        feats[organ] = d
    return feats
