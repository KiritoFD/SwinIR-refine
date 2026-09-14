"""L1 / offset-aligned L1 (mimicked alignment) + metrics. No UWCL/CX."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_y(img):
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    y = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return y.squeeze(0) if squeeze else y


class OffsetAlignedLoss(nn.Module):
    """
    Train-only: predict bounded residual offsets, warp HR toward SR, then

        L = (1-α)‖sr − W(hr, off)‖₁ + α ‖sr − hr‖₁ + λ‖off‖₁

    Zero-init offsets → starts as pure L1. Inference graph unchanged.
    """

    def __init__(self, max_shift=3.0, w_pure=0.25, w_off=0.01, hidden=32):
        super().__init__()
        self.max_shift = max_shift
        self.w_pure = w_pure
        self.w_off = w_off
        self.offset_net = nn.Sequential(
            nn.Conv2d(6, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, 1, 1),
        )
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def _warp(self, img, offset_px):
        B, _, H, W = img.shape
        ys = torch.linspace(-1.0, 1.0, H, device=img.device, dtype=img.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=img.device, dtype=img.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((gx, gy), 0).unsqueeze(0).expand(B, -1, -1, -1)
        scale = torch.empty(B, 2, H, W, device=img.device, dtype=img.dtype)
        scale[:, 0] = 2.0 / max(W - 1, 1)
        scale[:, 1] = 2.0 / max(H - 1, 1)
        grid = (base + offset_px * scale).permute(0, 2, 3, 1)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, sr, hr, log_var=None):
        sr, hr = sr.float(), hr.float()
        off = torch.tanh(self.offset_net(torch.cat([sr, hr], 1))) * self.max_shift
        hr_aln = self._warp(hr, off)
        l1_aln = (sr - hr_aln).abs().mean()
        l1_pure = (sr - hr).abs().mean()
        l1_off = off.abs().mean()
        total = (1.0 - self.w_pure) * l1_aln + self.w_pure * l1_pure + self.w_off * l1_off
        parts = {"l1": l1_pure, "l1_aln": l1_aln, "off": l1_off, "total": total, "plain": l1_pure}
        return total, parts


def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr.clamp(0, 1), hr.clamp(0, 1))
    if float(mse) == 0:
        return 99.0
    return 10 * torch.log10(max_val**2 / mse)
