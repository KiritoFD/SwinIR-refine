"""Physical-informed supervisory signals for real-world optical zoom super-resolution.

Addresses the three physical bottlenecks of RealSR:
1. LL-Swap Photometric-Invariant Supervision: eliminates DC offset & white-balance drift.
2. Bounded Sub-Pixel Parallax-Tolerant Alignment: eliminates 0.3-0.5px registration wall.
3. Training-time Optical Consistency Loss: internalizes the forward PSF degradation operator.
4. Chromatic Aberration Collinearity Loss: suppresses purple fringing & color dispersion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ll_swap_target(lr_up: torch.Tensor, hr: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Photometric-Invariant target: LL subband from lr_up + HF subbands from hr.
    
    Eliminates aperture/lens radiometric DC-offset and white-balance drift penalty
    from the spatial L1 loss, while keeping 100% of high-frequency ground truth details.
    """
    from .wavelet import haar_dwt2, haar_iwt2
    B, C, H, W = hr.shape
    ph = H % 2
    pw = W % 2
    if ph or pw:
        hr = F.pad(hr, (0, pw, 0, ph), mode="replicate")
        lr_up = F.pad(lr_up, (0, pw, 0, ph), mode="replicate")
    w_hr = haar_dwt2(hr)
    w_lr = haar_dwt2(lr_up)
    # Replace LL (first C channels) of w_hr with LL of w_lr
    w_swapped = torch.cat([w_lr[:, :C], w_hr[:, C:]], dim=1)
    target = haar_iwt2(w_swapped)
    if ph or pw:
        target = target[:, :, :H, :W]
    return target.clamp(0.0, 1.0) if clamp else target


def bounded_align_target(pred: torch.Tensor, hr: torch.Tensor, max_shift: float = 0.5) -> torch.Tensor:
    """Searches for bounded sub-pixel shift delta in [-max_shift, max_shift]^2 minimizing L1 error.
    
    Fights dual-lens non-rigid parallax (0.3-0.5px registration wall) by finding the optimal
    sub-pixel grid alignment for each patch, eliminating contradictory edge gradients.
    """
    B, C, H, W = hr.shape
    s = float(max_shift)
    shifts = [
        (0.0, 0.0),
        (s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s),
        (s * 0.707, s * 0.707), (-s * 0.707, s * 0.707),
        (s * 0.707, -s * 0.707), (-s * 0.707, -s * 0.707),
    ]
    base_grid_y, base_grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=hr.device, dtype=hr.dtype),
        torch.linspace(-1, 1, W, device=hr.device, dtype=hr.dtype),
        indexing="ij"
    )
    base_grid = torch.stack([base_grid_x, base_grid_y], dim=-1).unsqueeze(0).expand(B, H, W, 2)

    best_l1 = None
    best_shifted = hr

    for dx, dy in shifts:
        ndx = 2.0 * dx / max(W - 1, 1)
        ndy = 2.0 * dy / max(H - 1, 1)
        grid = base_grid.clone()
        grid[..., 0] += ndx
        grid[..., 1] += ndy
        shifted = F.grid_sample(hr, grid, mode="bilinear", padding_mode="border", align_corners=True)
        l1 = (pred - shifted).abs().mean(dim=(1, 2, 3))
        if best_l1 is None:
            best_l1 = l1
            best_shifted = shifted
        else:
            mask = (l1 < best_l1).view(B, 1, 1, 1)
            best_shifted = torch.where(mask, shifted, best_shifted)
            best_l1 = torch.minimum(best_l1, l1)

    return best_shifted.detach()


def optical_consistency_loss(pred: torch.Tensor, lr: torch.Tensor, sigma: float = 0.6) -> torch.Tensor:
    """Forward optical cycle loss: enforces physical degradation consistency during training.
    
    Simulates optical PSF blur + camera downsampling, comparing against raw LR measurement.
    """
    device = pred.device
    dtype = pred.dtype
    if sigma > 0:
        ksize = max(int(round(sigma * 4)) | 1, 3)
        coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        k = (g[:, None] * g[None, :])
        k = (k / k.sum()).view(1, 1, ksize, ksize).expand(3, 1, ksize, ksize)
        pad_k = ksize // 2
        blurred = F.conv2d(F.pad(pred, (pad_k, pad_k, pad_k, pad_k), mode="replicate"), k, groups=3)
        est_lr = F.interpolate(blurred, size=lr.shape[-2:], mode="bicubic", align_corners=False)
    else:
        est_lr = F.interpolate(pred, size=lr.shape[-2:], mode="bicubic", align_corners=False)
    return F.l1_loss(est_lr, lr)


def chromatic_aberration_loss(pred: torch.Tensor) -> torch.Tensor:
    """Cross-Channel Gradient Collinearity Loss.
    
    Suppresses purple fringing and lateral chromatic aberration by encouraging
    R and B gradient vectors to be collinear with G (or luminance) gradient vectors.
    """
    device = pred.device
    dtype = pred.dtype
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()

    gx = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode="replicate"), kx.expand(3, 1, 3, 3), groups=3)
    gy = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode="replicate"), ky.expand(3, 1, 3, 3), groups=3)

    g_mag = torch.sqrt(gx[:, 1:2] ** 2 + gy[:, 1:2] ** 2 + 1e-6)

    loss = 0.0
    for c in [0, 2]:
        c_mag = torch.sqrt(gx[:, c:c+1] ** 2 + gy[:, c:c+1] ** 2 + 1e-6)
        cross_x = g_mag * gx[:, c:c+1] - c_mag * gx[:, 1:2]
        cross_y = g_mag * gy[:, c:c+1] - c_mag * gy[:, 1:2]
        loss = loss + (cross_x.abs() + cross_y.abs()).mean()
    return loss


def rgb_to_ycbcr_601(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor in [0, 1] to ITU-R BT.601 YCbCr.
    Y in [0, 1], Cb, Cr in [-0.5, 0.5].
    Matches official A-Strict evaluation specification.
    """
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


def chroma_tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total Variation loss on Cb and Cr channels.
    Suppresses sensor read noise / color mottling without blurring luminance Y edges.
    """
    ycbcr = rgb_to_ycbcr_601(x)
    cb = ycbcr[:, 1:2]
    cr = ycbcr[:, 2:3]

    dh_cb = (cb[:, :, 1:, :] - cb[:, :, :-1, :]).abs().mean()
    dw_cb = (cb[:, :, :, 1:] - cb[:, :, :, :-1]).abs().mean()
    dh_cr = (cr[:, :, 1:, :] - cr[:, :, :-1, :]).abs().mean()
    dw_cr = (cr[:, :, :, 1:] - cr[:, :, :, :-1]).abs().mean()

    return (dh_cb + dw_cb + dh_cr + dw_cr) * 0.5


def ycbcr_weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                           y_weight: float = 1.0, chroma_weight: float = 0.2) -> torch.Tensor:
    """L1 loss in BT.601 YCbCr space with asymmetric weighting.
    Focuses gradient energy on the evaluation channel Y while gently supervising Cb/Cr.
    """
    pred_ycbcr = rgb_to_ycbcr_601(pred)
    target_ycbcr = rgb_to_ycbcr_601(target)

    loss_y = (pred_ycbcr[:, 0:1] - target_ycbcr[:, 0:1]).abs().mean()
    loss_cb = (pred_ycbcr[:, 1:2] - target_ycbcr[:, 1:2]).abs().mean()
    loss_cr = (pred_ycbcr[:, 2:3] - target_ycbcr[:, 2:3]).abs().mean()

    return y_weight * loss_y + chroma_weight * (loss_cb + loss_cr)

