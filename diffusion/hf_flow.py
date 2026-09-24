"""Residual high-frequency rectified flow on top of the regression SR (Direction C).

Why this and not full-image diffusion
-------------------------------------
The whole project's negative diffusion result (every flow arm < bicubic on PSNR) was
measured under a PSNR-dominated lens.  But under the current criteria (SSIM/MUSIQ/
MANIQA primary) a generative model's *strength* is exactly the high-frequency detail
that an L1 (+wavelet-HF) regressor cannot recover -- L1 gives the conditional MEAN of
the HF, i.e. blur.  So we keep everything the regression is good at (the LL band and
the HF mean, which carry PSNR/SSIM) and model ONLY the residual HF distributionally:

    DWT(pred) -> [LL_r, HF_r]      (regression: good LL, smeared HF mean)
    DWT(hr)   -> [LL_t, HF_t]      (target)
    flow head generates  x0 = HF_t - HF_r   conditioned on cond=[LL_r, HF_r]
    HF_final  = HF_r + sample()     (a small realistic correction, not a rewrite)
    out       = IWT([LL_r, HF_final])

Because the LL and the HF *mean* come straight from the regressor, PSNR/SSIM can't
collapse the way a from-noise full-image flow did; the head only adds the unobservable
detail where perception lives.  Reuses flow.py (rectified flow) + wavelet.py (Haar).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .flow import flow_loss, sample_flow
from .unet import build_unet
from .wavelet import haar_dwt2, haar_iwt2

HF_BANDS = 3  # HL, LH, HH -> 3C channels after split


def split_bands(bands: torch.Tensor):
    """[LL,HL,LH,HH] concat (B,4C,h/2,w/2) -> LL (B,C,..), HF (B,3C,..)."""
    C = bands.shape[1] // 4
    return bands[:, :C], bands[:, C:]


def merge_bands(ll: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
    return torch.cat([ll, hf], dim=1)


def build_hf_head(size: str = "XS", base: int = 64, ch: int = 3, **kw) -> torch.nn.Module:
    """Flow net over the HF residual (3C channels) conditioned on [LL_r, HF_r] (4C)."""
    in_channels = HF_BANDS * ch + 4 * ch          # x_t + cond
    kw.setdefault("mult", (1, 2, 4, 4))
    kw.setdefault("num_res", 2)
    kw.setdefault("attn_levels", (2, 3))
    kw.setdefault("in_stride", 1)
    kw.setdefault("out_scale", 1)
    return build_unet(size, base=base, in_channels=in_channels,
                      out_channels=HF_BANDS * ch, **kw)


def hf_flow_training_loss(head, pred, hr):
    """One rectified-flow loss on the HF residual; returns (loss, diag)."""
    dp, dt = haar_dwt2(pred), haar_dwt2(hr)
    ll_r, hf_r = split_bands(dp)
    _, hf_t = split_bands(dt)
    cond = torch.cat([ll_r, hf_r], dim=1)
    x0 = hf_t - hf_r                                  # the detail the regressor missed
    loss, info = flow_loss(head, x0, cond=cond)       # velocity MSE
    return loss, {"x0_absmean": x0.abs().mean().item()}


@torch.no_grad()
def hf_refine(head, pred, steps: int = 4, device="cuda", seed: int = 0, scale: float = 1.0):
    """Regression prediction (B,3,H,W) -> HF-refined prediction, same shape.

    The residual is generated from noise conditioned on the regression's LL/HF mean;
    `scale` lets one dial back how much generated detail is added (perception/PSNR).
    """
    dp = haar_dwt2(pred)
    ll_r, hf_r = split_bands(dp)
    ch = pred.shape[1]
    shape = (pred.shape[0], HF_BANDS * ch, hf_r.shape[2], hf_r.shape[3])
    x0 = sample_flow(head, shape, cond=torch.cat([ll_r, hf_r], dim=1),
                     steps=steps, solver="heun", device=device, seed=seed)
    hf_final = (hf_r + scale * x0).clamp(-2.0, 2.0)
    out = haar_iwt2(merge_bands(ll_r, hf_final))
    # return unclamped float: the caller (to_u8) does the [0,1] range clip; clamping
    # here would perturb the LL band, which the flow must leave untouched by construction.
    return out


def _smoke():
    """Shape / finiteness / gradient / energy self-check (no data, no ckpt)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 128, 128, device=dev)
    hr = torch.rand(2, 3, 128, 128, device=dev)
    head = build_hf_head("XS", base=32, ch=3).to(dev)
    loss, diag = hf_flow_training_loss(head, pred, hr)
    assert torch.isfinite(loss), "hf flow loss not finite"
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    assert g > 0, "no grad to the hf head"
    for p in head.parameters():
        p.grad = None
    refined = hf_refine(head, pred, steps=2, device=dev, seed=1)
    assert refined.shape == pred.shape, f"refine shape {refined.shape}"
    assert torch.isfinite(refined).all(), "refine produced non-finite"
    # LL is untouched by construction -> low-frequency fidelity preserved
    ll0 = split_bands(haar_dwt2(pred))[0]
    ll1 = split_bands(haar_dwt2(refined))[0]
    assert (ll0 - ll1).abs().max().item() < 1e-3, "hf_refine leaked into the LL band"
    print(f"hf_flow smoke OK: loss={loss.item():.4f} x0_absmean={diag['x0_absmean']:.4f} "
          f"grad={g:.1f} refined finite + LL preserved ({(ll0-ll1).abs().max().item():.1e})")


if __name__ == "__main__":
    _smoke()
