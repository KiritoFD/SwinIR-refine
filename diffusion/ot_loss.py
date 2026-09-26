"""Optimal-Transport losses for RealSR — attack the point-to-point / registration wall.

L1/L2 penalise pixels at the SAME coordinate, so a 0.3 px misalignment (irreducible in
RealSR zoom pairs — see FINAL_REPORT 附录F) forces the net toward the conditional mean
(blur).  An OT distance compares two *distributions* of local patches instead: a sharp
texture that is merely shifted costs little, so the net can render real high-frequency
without the point-to-point penalty.  This directly trades a little PSNR/SSIM for much
higher no-reference perceptual quality (MUSIQ/MANIQA) — the perception variants we want.

Two differentiable OT implementations:
  * SWD  — sliced Wasserstein: project patch clouds onto random 1-D directions, match
           sorted quantiles.  O(M log M), stable, our default.
  * Sinkhorn — entropic-regularised OT between subsampled patch clouds.  O(M^2 * iters).

Can be applied on raw image patches (ot_on="image") or on the orthonormal wavelet
high-frequency subbands (ot_on="wavelet") to combine with our winning wavelet-HF idea.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B,C,H,W) -> (M, C*p*p) non-overlapping patches; pads to a multiple of p."""
    B, C, H, W = x.shape
    ph = (H // p) * p
    pw = (W // p) * p
    x = x[:, :, :ph, :pw]
    x = x.reshape(B, C, ph // p, p, pw // p, p).permute(0, 2, 3, 4, 5, 1)
    return x.reshape(-1, C * p * p)


def sliced_wasserstein(pred, target, patch=8, n_proj=256, seed=0):
    a = _patchify(pred.float(), patch)
    b = _patchify(target.float(), patch)
    d = a.shape[1]
    # deterministic random directions (fixed per forward keeps gradient low-variance)
    g = torch.Generator(device=a.device).manual_seed(seed)
    proj = torch.randn(d, n_proj, device=a.device, generator=g)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-6)
    pa = (a @ proj)           # (M, n_proj)
    pb = (b @ proj)
    sa = pa.sort(dim=0).values
    sb = pb.sort(dim=0).values
    return (sa - sb).abs().mean()


def sinkhorn(pred, target, patch=8, eps=0.05, iters=10, n=512, seed=0, p_order=2):
    a = _patchify(pred.float(), patch)
    b = _patchify(target.float(), patch)
    g = torch.Generator(device=a.device).manual_seed(seed)
    if a.shape[0] > n:
        a = a[torch.randperm(a.shape[0], device=a.device, generator=g)[:n]]
    if b.shape[0] > n:
        b = b[torch.randperm(b.shape[0], device=b.device, generator=g)[:n]]
    C = torch.cdist(a, b, p=p_order) ** p_order          # (M,M) cost
    C = C / C.mean().clamp_min(1e-6)
    f = torch.zeros(C.shape[0], device=a.device)
    h = torch.zeros(C.shape[1], device=a.device)
    K = torch.exp(-C / eps)
    for _ in range(iters):                              # log-domain Sinkhorn
        f = -eps * torch.logsumexp(torch.log(K).T + h[None, :], dim=1)
        h = -eps * torch.logsumexp(torch.log(K) + f[:, None], dim=1)
    P = torch.exp((f[:, None] + h[None, :] - C) / eps)  # plan
    marg = 1.0 / P.shape[0]
    return (P * C).sum() * marg


def hf_subbands(x, levels=2):
    """concat Haar high-frequency subbands across scales (orthonormal), each resampled
    to the level-1 HF grid so channels align, for OT-on-wavelet."""
    from .wavelet import haar_dwt2
    hf = []
    ref = None
    cur = x.float()
    for _ in range(levels):
        if cur.shape[-1] % 2 or cur.shape[-2] % 2 or cur.shape[-1] < 8 or cur.shape[-2] < 8:
            break
        d = haar_dwt2(cur)
        C = d.shape[1] // 4
        band = d[:, C:]
        if ref is None:
            ref = band
        else:
            band = F.interpolate(band, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        hf.append(band)
        cur = d[:, :C]
    return torch.cat(hf, dim=1) if hf else x.float()


def ot_loss(pred, target, kind="swd", on="image", patch=8, n_proj=256, seed=0):
    if on == "wavelet":
        pred, target = hf_subbands(pred), hf_subbands(target)
    if kind == "sinkhorn":
        return sinkhorn(pred, target, patch=patch, seed=seed)
    return sliced_wasserstein(pred, target, patch=patch, n_proj=n_proj, seed=seed)
