"""Grad-CAM heatmaps for Merlin and Pillar-0 — manual implementation.

For the production Phase 2B concat probe, use `ensure_concat_heatmap()`:
it splits the 3200-d weight row into Merlin (W[:, :2048]) and Pillar-0
(W[:, 2048:]) slots, runs Grad-CAM with an L2-norm wrap to mirror the
inference-time forward, caches the NIfTI under heatmaps/<encoder>/<sid>/.


Vanilla CAM (per-voxel = W[f] · feature[:,v]) is mathematically exact ONLY when
the head ends in average-pool — i.e. Merlin's AdaptiveAvgPool3d. For Pillar-0,
whose emb is the L2-normalised concat of three per-scale max-pools, vanilla CAM
is an approximation that produces low-signal noise (the visible "regular dot
pattern" in the earlier heatmaps was the 64^3 token grid showing through the
L2-norm mismatch). Grad-CAM handles arbitrary head non-linearity via gradients.

Recipe (Selvaraju et al. 2017, applied per encoder):
    A    = capture target layer output during forward (forward hook)
    g_A  = backprop d(logit_f)/d(A) via tensor.retain_grad()
    w_c  = global_avg_pool_over_spatial(g_A[:, c, ...])
    cam  = ReLU(sum_c w_c · A[:, c, ...])
returning a single-channel 3D heatmap at the target layer's spatial resolution.

We initially tried Captum's `LayerGradCam` but hit two encoder-specific issues:
  - Merlin's I3D-ResNet uses torch.utils.checkpoint internally; Captum's
    captured activation gets detached from the autograd graph between forward
    and backward, raising "One of the differentiated Tensors appears to not
    have been used in the graph".
  - Pillar-0's maxpool is reachable via dotted-attribute access but
    nn.Module.__setattr__ rejects assigning a plain callable for monkey-patch.

Hand-rolled hook-based Grad-CAM avoids both. Same math, ~25 lines per encoder.

Axis fix from `9fcb713` (Merlin layer4 is in (S, R, A) order) carries through:
Grad-CAM's output has the same spatial axis order as the target layer's output,
so we still need the (1, 2, 0) permute before upsampling.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import merlinplus as mp


# --------------------------------------------------------------------------- #
# Probe I/O — unchanged
# --------------------------------------------------------------------------- #

def load_probe(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (W, b, findings) from a torch checkpoint saved by scripts/33."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    W = sd["weight"].numpy()                       # (n_findings, dim)
    b = sd.get("bias", torch.zeros(W.shape[0])).numpy()
    findings = ck["findings"]
    return W, b, findings


def finding_row(W: np.ndarray, findings: list[str], finding: str) -> np.ndarray:
    if finding not in findings:
        raise KeyError(f"finding {finding!r} not in probe (have {len(findings)})")
    return W[findings.index(finding)]


# --------------------------------------------------------------------------- #
# Shared Grad-CAM compute: forward hook → activation, backward → gradient,
# spatial-avg gradient as channel weights, ReLU sum of weight × activation.
# --------------------------------------------------------------------------- #

def _gradcam_compute(target_layer, model_fn, target_idx: int) -> torch.Tensor:
    """Run model_fn(), capturing target_layer output (forward hook) and the
    gradient w.r.t. that output (backward hook). Compute Grad-CAM.

    Using two separate hooks (forward + full_backward) instead of `retain_grad`
    handles models with gradient checkpointing (Merlin's I3D-ResNet uses it):
    the captured activation can have requires_grad=False at forward time and
    the backward hook still fires correctly via the module-level callback.

    Returns (*spatial) shape after channel-summed ReLU(weight × activation).
    """
    fwd: dict = {}
    bwd: dict = {}

    def fwd_hook(_m, _inp, output):
        if "A" not in fwd:
            fwd["A"] = output.detach().clone()

    def bwd_hook(_m, _grad_input, grad_output):
        if "grad" not in bwd:
            bwd["grad"] = grad_output[0].detach().clone()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model_fn()                  # (1, n_findings)
        logit = logits[0, target_idx]
        logit.backward()
    finally:
        h1.remove(); h2.remove()

    if "A" not in fwd or "grad" not in bwd:
        raise RuntimeError("hook didn't capture activation or gradient")
    A = fwd["A"]                             # (1, C, *spatial)
    grad = bwd["grad"]                       # (1, C, *spatial)
    spatial_dims = tuple(range(2, grad.dim()))
    weights = grad.mean(dim=spatial_dims, keepdim=True)
    cam = (weights * A).sum(dim=1)           # (1, *spatial)
    return F.relu(cam).squeeze(0)            # (*spatial)


# --------------------------------------------------------------------------- #
# Merlin Grad-CAM
# --------------------------------------------------------------------------- #

def _merlin_cam_inner(study_id: str, finding_idx: int, model,
                      W_t: torch.Tensor, b_t: torch.Tensor, *,
                      apply_l2: bool) -> tuple[np.ndarray, np.ndarray]:
    """Shared Merlin Grad-CAM body; preprocess + hook compute + axis fix + upsample.

    `apply_l2`: when True, L2-normalises the 2048-d embedding before the linear
    head. Concat-probe path. Backprop through the L2 Jacobian is handled by
    autograd automatically.
    """
    from src.embeddings import merlin as M

    ct_img = mp.load_ct(study_id)
    if ct_img is None:
        raise FileNotFoundError(f"CT not found: {study_id}")
    ct_img = nib.as_closest_canonical(ct_img)
    ct_shape = ct_img.shape
    ct_affine = ct_img.affine.copy()
    spacing = tuple(float(abs(ct_affine[i, i])) for i in range(3))

    vol_np = np.asanyarray(ct_img.dataobj).astype(np.float32)
    t = torch.from_numpy(vol_np).cuda()
    factors = tuple(s / tg for s, tg in zip(spacing, M.TARGET_SPACING))
    new_shape = tuple(max(1, int(round(d * f))) for d, f in zip(t.shape, factors))
    t_res = F.interpolate(t[None, None], size=new_shape, mode="trilinear",
                          align_corners=False)[0, 0]
    pads = []
    for d_in, d_t in zip(reversed(t_res.shape), reversed(M.TARGET_SHAPE)):
        diff = max(0, d_t - d_in)
        pads += [diff // 2, diff - diff // 2]
    t_pad = F.pad(t_res, pads, mode="constant", value=0.0) if any(pads) else t_res
    crop_offsets, slices = [], []
    for d_in, d_t in zip(t_pad.shape, M.TARGET_SHAPE):
        start = max(0, (d_in - d_t) // 2)
        crop_offsets.append(start)
        slices.append(slice(start, start + d_t))
    t_pp = t_pad[tuple(slices)]
    t_pp.clamp_(M.HU_RANGE[0], M.HU_RANGE[1])
    t_pp = (t_pp - M.HU_RANGE[0]) / (M.HU_RANGE[1] - M.HU_RANGE[0])

    target_layer = model.model.encode_image.i3_resnet.layer4

    def model_fn():
        x = t_pp[None, None].requires_grad_()
        out = model(x)                                  # (1, 1, 2048)
        emb = out.float().reshape(out.shape[0], -1)     # (1, 2048)
        if apply_l2:
            emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return F.linear(emb, W_t, b_t)                   # (1, n_findings)

    with torch.set_grad_enabled(True):
        cam_lf = _gradcam_compute(target_layer, model_fn, finding_idx)
    cam_lf_ras = cam_lf.permute(1, 2, 0).contiguous()
    cam_pp = F.interpolate(cam_lf_ras[None, None].float(),
                           size=M.TARGET_SHAPE,
                           mode="trilinear", align_corners=False)[0, 0]

    pad_shape = tuple(t_pad.shape)
    full_pp = torch.zeros(pad_shape, dtype=cam_pp.dtype, device=cam_pp.device)
    inv_slices = [slice(crop_offsets[i], crop_offsets[i] + M.TARGET_SHAPE[i]) for i in range(3)]
    full_pp[tuple(inv_slices)] = cam_pp
    if any(pads):
        per_dim = []
        for i, d_in in enumerate(t_res.shape):
            rev_i = len(t_res.shape) - 1 - i
            left = pads[2 * rev_i]
            per_dim.append(slice(left, left + d_in))
        cam_res = full_pp[tuple(per_dim)]
    else:
        cam_res = full_pp

    cam_ct = F.interpolate(cam_res[None, None].float(), size=ct_shape,
                           mode="trilinear", align_corners=False)[0, 0]
    return cam_ct.cpu().numpy(), ct_affine


def merlin_cam(study_id: str, finding: str, model,
               probe_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Generate Merlin Grad-CAM for one (study, finding) — legacy CAM-probe path
    (no L2-norm before the linear head)."""
    W, b, findings = load_probe(probe_path)
    finding_idx = findings.index(finding)
    W_t = torch.from_numpy(W.astype(np.float32)).cuda()
    b_t = torch.from_numpy(b.astype(np.float32)).cuda()
    return _merlin_cam_inner(study_id, finding_idx, model, W_t, b_t, apply_l2=False)


# --------------------------------------------------------------------------- #
# Pillar-0 Grad-CAM (no monkey-patch — register forward+backward hooks instead)
# --------------------------------------------------------------------------- #

def _pillar0_cam_inner(study_id: str, finding_idx: int, model,
                       W_t: torch.Tensor, b_t: torch.Tensor, *,
                       apply_l2: bool) -> tuple[np.ndarray, np.ndarray]:
    """Shared Pillar-0 Grad-CAM body. Pillar-0 emb is already L2≈1 by design,
    so the apply_l2 wrap is mostly a no-op but kept for symmetry / safety."""
    from src.embeddings import pillar0 as P

    ct_img = mp.load_ct(study_id)
    if ct_img is None:
        raise FileNotFoundError(f"CT not found: {study_id}")
    ct_img = nib.as_closest_canonical(ct_img)
    ct_shape = ct_img.shape
    ct_affine = ct_img.affine.copy()

    vol = np.asanyarray(ct_img.dataobj).astype(np.float32)
    vol = np.ascontiguousarray(np.transpose(vol, (2, 1, 0)))            # (S, A, R)
    t = torch.from_numpy(vol)[None, None].cuda()
    t = F.interpolate(t, size=(P.SIZE,) * 3, mode="trilinear", align_corners=False)[0, 0]
    x = P._windowing()(t, "all", "CT").unsqueeze(0)                     # (1, 11, 384, 384, 384)

    fwd: dict = {}
    bwd: dict = {}

    def fwd_hook(_m, inputs, _output):
        if "A" not in fwd:
            fwd["A"] = inputs[0].detach().clone()

    def bwd_hook(_m, grad_input, _grad_output):
        if "grad" not in bwd:
            bwd["grad"] = grad_input[0].detach().clone()

    target = model.model.visual.maxpool
    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)
    try:
        with torch.set_grad_enabled(True):
            emb = model.extract_vision_feats(image={P.MODALITY: x}).float()
            if apply_l2:
                emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
            logits = F.linear(emb, W_t, b_t)
            logit = logits[0, finding_idx]
            logit.backward()
    finally:
        h1.remove(); h2.remove()

    if "A" not in fwd or "grad" not in bwd:
        raise RuntimeError("Pillar-0 hooks didn't capture")
    A = fwd["A"]
    grad = bwd["grad"]
    spatial_dims = tuple(range(2, grad.dim()))
    weights = grad.mean(dim=spatial_dims, keepdim=True)
    cam_flat = (weights * A).sum(dim=1).squeeze(0)
    cam_flat = F.relu(cam_flat)
    if cam_flat.dim() == 1:
        n = cam_flat.numel()
        g = round(n ** (1 / 3))
        cam_g = cam_flat.reshape(g, g, g)
    else:
        cam_g = cam_flat
    cam_full = F.interpolate(cam_g[None, None].float(), size=(P.SIZE,) * 3,
                             mode="trilinear", align_corners=False)[0, 0]
    cam_ras = cam_full.permute(2, 1, 0).contiguous()
    cam_ct = F.interpolate(cam_ras[None, None].float(), size=ct_shape,
                           mode="trilinear", align_corners=False)[0, 0]
    return cam_ct.cpu().numpy(), ct_affine


def pillar0_cam(study_id: str, finding: str, model,
                probe_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Pillar-0 Grad-CAM — legacy CAM-probe path (no L2-norm)."""
    W, b, findings = load_probe(probe_path)
    finding_idx = findings.index(finding)
    W_t = torch.from_numpy(W.astype(np.float32)).cuda()
    b_t = torch.from_numpy(b.astype(np.float32)).cuda()
    return _pillar0_cam_inner(study_id, finding_idx, model, W_t, b_t, apply_l2=False)


# --------------------------------------------------------------------------- #
# Auto-generation via the production concat probe (Phase 2B)
# --------------------------------------------------------------------------- #

HEAT_ROOT = Path("/mnt/e/ctvlm/heatmaps")


@__import__("functools").lru_cache(maxsize=1)
def _concat_probe_slots() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the concat probe once; return (W_M, W_P, b, findings)."""
    from src.config import paths
    ck = torch.load(paths.checkpoints_dir / "concat_rate_probe.pt",
                    map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    W = sd["weight"].numpy()                                # (220, 3200)
    b = sd.get("bias", torch.zeros(W.shape[0])).numpy()
    findings = ck["findings"]
    return W[:, :2048], W[:, 2048:], b, findings


def ensure_concat_heatmap(study_id: str, encoder: str, finding: str,
                          *, heat_root: Path = HEAT_ROOT,
                          verbose: bool = True) -> Path | None:
    """Lazy heatmap cache: generate via the concat probe slot if not on disk.

    Returns the on-disk Path on success, None on failure / unknown finding.
    Per-encoder forward + CAM is ~3 s on a 3090; model load is amortised via
    each encoder's own lru_cache.
    """
    out = heat_root / encoder / study_id / f"{finding}.nii.gz"
    if out.exists():
        return out

    W_M, W_P, b, findings = _concat_probe_slots()
    if finding not in findings:
        return None
    finding_idx = findings.index(finding)
    b_t = torch.from_numpy(b.astype(np.float32)).cuda()

    try:
        if encoder == "merlin":
            from src.embeddings import merlin as M
            model = M.load_model()
            W_t = torch.from_numpy(W_M.astype(np.float32)).cuda()
            h, aff = _merlin_cam_inner(study_id, finding_idx, model, W_t, b_t,
                                       apply_l2=True)
        elif encoder == "pillar0":
            from src.embeddings import pillar0 as P
            model = P.load_model()
            W_t = torch.from_numpy(W_P.astype(np.float32)).cuda()
            h, aff = _pillar0_cam_inner(study_id, finding_idx, model, W_t, b_t,
                                        apply_l2=True)
        else:
            return None
    except Exception as e:                # noqa: BLE001 — log + return None
        if verbose:
            print(f"  [cam.ensure] {encoder}/{study_id}/{finding} FAILED: "
                  f"{type(e).__name__}: {str(e)[:120]}")
        return None

    save_heatmap(h, aff, out, study_id=study_id)
    if verbose:
        print(f"  [cam.ensure] generated {encoder}/{study_id}/{finding}")
    return out


# --------------------------------------------------------------------------- #
# Post-processing — body mask + Gaussian smoothing + clip+normalize
# --------------------------------------------------------------------------- #

def _body_mask_from_ct(study_id: str, hu_threshold: float = -500.0) -> np.ndarray:
    """Boolean mask in canonical-RAS CT space — True where CT HU > threshold (tissue)."""
    ct_img = mp.load_ct(study_id)
    if ct_img is None:
        raise FileNotFoundError(f"CT not found: {study_id}")
    ct_img = nib.as_closest_canonical(ct_img)
    return np.asanyarray(ct_img.dataobj) > hu_threshold


def save_heatmap(heatmap: np.ndarray, affine: np.ndarray, out_path: str | Path,
                 *,
                 study_id: str | None = None,
                 take_abs: bool = False,
                 body_mask_hu: float | None = -500.0,
                 smooth_sigma_vox: float = 2.0,
                 percentile_clip: tuple[float, float] | None = (1.0, 99.0)) -> Path:
    """Save heatmap as a NIfTI aligned to CT; clean up artefacts before saving."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = heatmap.astype(np.float32)

    if take_abs:
        h = np.abs(h)

    mask = None
    if body_mask_hu is not None and study_id:
        mask = _body_mask_from_ct(study_id, hu_threshold=body_mask_hu)
        if mask.shape != h.shape:
            print(f"  [save_heatmap] body mask shape mismatch {mask.shape} vs {h.shape}; skipping mask")
            mask = None
        else:
            h = h * mask.astype(np.float32)

    if smooth_sigma_vox and smooth_sigma_vox > 0:
        from scipy.ndimage import gaussian_filter
        h = gaussian_filter(h, sigma=smooth_sigma_vox, mode="constant", cval=0.0)

    if percentile_clip is not None:
        nz = h[h != 0]
        if nz.size > 0:
            lo, hi = np.percentile(nz, percentile_clip)
            h = np.clip(h, lo, hi)

    rng = h.max() - h.min()
    if rng > 1e-6:
        h = (h - h.min()) / rng
        if mask is not None:
            h = h * mask.astype(np.float32)

    nib.save(nib.Nifti1Image(h.astype(np.float32), affine), str(out_path))
    return out_path
