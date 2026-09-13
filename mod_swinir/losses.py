"""RealSR-oriented losses: residual-safe L1 + local amp + confidence-weighted HF."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast_off(x):
    return torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=False)


def highpass(x, ksize=5, sigma=1.2):
    """HP(x) = x − G_σ * x  (per-channel Gaussian)."""
    device = x.device
    dtype = x.dtype
    coords = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    C = x.shape[1]
    kernel = kernel.expand(C, 1, ksize, ksize).contiguous()
    pad = ksize // 2
    return x - F.conv2d(x, kernel, padding=pad, groups=C)


def amplitude_spectrum_loss(sr, hr, eps=1e-6):
    """Global |F| L1 — misalignment-robust (shift-invariant amplitude)."""
    with _autocast_off(sr):
        s, h = sr.float(), hr.detach().float()
        hs, ws = s.shape[-2:]
        a_s = torch.abs(torch.fft.rfft2(s, norm="ortho"))
        a_h = torch.abs(torch.fft.rfft2(h, s=(hs, ws), norm="ortho"))
        return (a_s - a_h).abs().mean()


def patch_amplitude_spectrum_loss(sr, hr, patch=32, stride=16):
    """
    Local (patch-wise) amplitude loss.

    Real degradation + residual misregistration are *local* (PSF varies across
    the field; T_τ is spatially varying). Full-image FFT averages that away.
    Unfold HR resolution into overlapping patches, match |F| per patch:
        L = mean_w  ‖ |F_w(ŷ)| − |F_w(y)| ‖_1
    """
    with _autocast_off(sr):
        s = sr.float()
        h = hr.detach().float()
        B, C, H, W = s.shape
        if H < patch or W < patch:
            return amplitude_spectrum_loss(s, h)

        def _patches(img):
            x = F.pad(img, (0, max(0, patch - W % stride if W % stride else 0), 0, 0))
            # simpler: unfold with explicit pad if needed
            ph = (patch - H % patch) % patch if H % patch else 0
            pw = (patch - W % patch) % patch if W % patch else 0
            if ph or pw:
                img = F.pad(img, (0, pw, 0, ph), mode="reflect")
            u = F.unfold(img, kernel_size=patch, stride=stride)  # B, C*p*p, L
            L = u.shape[-1]
            return u.view(B, C, patch, patch, L)

        ps = _patches(s)
        ph_ = _patches(h)
        L = ps.shape[-1]
        # batch patches into FFT: (B*L, C, p, p)
        ps = ps.permute(0, 4, 1, 2, 3).reshape(B * L, C, patch, patch)
        ph_ = ph_.permute(0, 4, 1, 2, 3).reshape(B * L, C, patch, patch)
        a_s = torch.abs(torch.fft.rfft2(ps, norm="ortho"))
        a_h = torch.abs(torch.fft.rfft2(ph_, norm="ortho"))
        return (a_s - a_h).abs().mean()


def local_confidence(sr, hr, win=15, sigma=2.0, eps=1e-3):
    """
    Pixel-wise alignment confidence in [0,1] from local NCC of high-pass fields.

    High-pass removes slowly varying bias; local NCC ≈ 1 when structures align,
    ≈ 0 when the patch is misregistered / dominated by uncorrelated noise.
    Used as a *detached* weight so the net cannot game confidence by blurring.
    """
    with _autocast_off(sr):
        s = highpass(sr.float().detach(), ksize=5, sigma=1.2)
        h = highpass(hr.float().detach(), ksize=5, sigma=1.2)
        # mean over channels → luminance-like
        s = s.mean(dim=1, keepdim=True)
        h = h.mean(dim=1, keepdim=True)
        k = win if win % 2 == 1 else win + 1
        pad = k // 2

        def box(x):
            return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k, stride=1)

        mu_s, mu_h = box(s), box(h)
        ss, hh, sh = box(s * s), box(h * h), box(s * h)
        var_s = (ss - mu_s * mu_s).clamp(min=0)
        var_h = (hh - mu_h * mu_h).clamp(min=0)
        cov = sh - mu_s * mu_h
        ncc = cov / (torch.sqrt(var_s * var_h) + eps)
        # map [-1,1] → [0,1], slightly sharpen
        conf = (0.5 * (ncc + 1.0)).clamp(0, 1)
        return conf  # B,1,H,W


def confidence_hf_loss(sr, hr, win=15, conf_power=1.0):
    """
    Confidence-weighted high-frequency L1:

        L = Σ_i c_i |HP(ŷ)_i − HP(y)_i|  /  (Σ_i c_i + ε)

    Low-confidence (misaligned / flaky) pixels contribute less, reducing the
    “blur toward multi-modal mean” pressure of unweighted pixel losses.
    """
    with _autocast_off(sr):
        c = local_confidence(sr, hr, win=win)
        if conf_power != 1.0:
            c = c.clamp(0, 1) ** conf_power
        err = (highpass(sr.float()) - highpass(hr.detach().float())).abs()
        # broadcast conf (B,1,H,W) over channels
        num = (err * c).sum()
        den = c.sum() * err.shape[1] + 1e-6
        return num / den


class RealSRLoss(nn.Module):
    """
    Main PSNR-track objective for RealSR:

        L = w_l1 * ‖ŷ−y‖_1
          + w_amp_patch * Σ_w ‖|F_w(ŷ)|−|F_w(y)|‖_1
          + w_hf_conf * c-weighted |HP(ŷ)−HP(y)|
          + (optional) global amp, plain grad

    All terms ≥ 0. Confidence map is detached.
    """

    def __init__(
        self,
        w_l1=1.0,
        w_amp_patch=0.05,
        w_hf_conf=0.25,
        w_amp_global=0.0,
        w_grad=0.0,
        patch=32,
        stride=16,
        conf_win=15,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_amp_patch = w_amp_patch
        self.w_hf_conf = w_hf_conf
        self.w_amp_global = w_amp_global
        self.w_grad = w_grad
        self.patch = patch
        self.stride = stride
        self.conf_win = conf_win

    def forward(self, sr, log_var, hr):
        sr_c = sr.float()
        hr = hr.float()
        parts = {}
        l1 = (sr_c - hr).abs().mean()
        total = self.w_l1 * l1
        parts["l1"] = l1

        if self.w_amp_patch > 0:
            ap = patch_amplitude_spectrum_loss(sr_c, hr, self.patch, self.stride)
            total = total + self.w_amp_patch * ap
            parts["amp_patch"] = ap

        if self.w_hf_conf > 0:
            hf = confidence_hf_loss(sr_c, hr, win=self.conf_win)
            total = total + self.w_hf_conf * hf
            parts["hf_conf"] = hf

        if self.w_amp_global > 0:
            ag = amplitude_spectrum_loss(sr_c, hr)
            total = total + self.w_amp_global * ag
            parts["amp_global"] = ag

        if self.w_grad > 0:
            def grads(x):
                return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (
                    x[:, :, 1:, :] - x[:, :, :-1, :]
                ).abs().mean()

            g = grads(sr_c - hr)
            total = total + self.w_grad * g
            parts["grad"] = g

        parts["total"] = total
        return total, parts


# --- back-compat names used by older train paths ---

def gradient_loss(sr, hr):
    def grads(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    sdx, sdy = grads(sr)
    hdx, hdy = grads(hr)
    return (sdx - hdx).abs().mean() + (sdy - hdy).abs().mean()


def contextual_loss(sr, hr, pool=4, bandwidth=0.5):
    device_type = "cuda" if sr.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        s = F.avg_pool2d(sr.float(), pool, pool)
        h = F.avg_pool2d(hr.detach().float(), pool, pool)
        B, C, H, W = s.shape
        xf = F.normalize(s.permute(0, 2, 3, 1).reshape(B, H * W, C), dim=-1)
        yf = F.normalize(h.permute(0, 2, 3, 1).reshape(B, H * W, C), dim=-1)
        d = (1.0 - torch.bmm(xf, yf.transpose(1, 2))).clamp(min=0.0, max=2.0)
        d_min, _ = d.min(dim=2, keepdim=True)
        d_norm = d / (d_min + 1e-8)
        w = torch.exp(((1.0 - d_norm) / bandwidth).clamp(min=-20.0, max=20.0))
        cx = (w.max(dim=2).values / (w.sum(dim=2) + 1e-8)).clamp(min=1e-6, max=1.0)
        return (-torch.log(cx)).mean()


class UWCLLoss(nn.Module):
    """Legacy wrapper — prefer RealSRLoss for PSNR track."""

    def __init__(self, w_uw=1.0, w_ctx=0.0, w_grad=0.0, w_plain=1.0, w_amp=0.0, sigma_eps=1e-2):
        super().__init__()
        self.inner = RealSRLoss(w_l1=w_plain, w_amp_patch=w_amp, w_hf_conf=0.0, w_grad=w_grad)
        self.w_ctx = w_ctx

    def forward(self, sr, log_var, hr):
        total, parts = self.inner(sr, log_var, hr)
        if self.w_ctx > 0:
            ctx = contextual_loss(sr, hr)
            total = total + self.w_ctx * ctx
            parts["ctx"] = ctx
        parts["total"] = total
        parts["uw_l1"] = parts.get("l1", total)
        parts["plain"] = parts.get("l1", total)
        return total, parts


def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr.clamp(0, 1), hr.clamp(0, 1))
    if mse.item() == 0:
        return 99.0
    return 10 * torch.log10(max_val**2 / mse)


def rgb_to_y(img):
    """BT.601 Y from RGB in [0,1]. Works on (B,3,H,W) or (3,H,W)."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.squeeze(0) if squeeze else y


class OffsetAlignedLoss(nn.Module):
    """
    Train-time mimicked alignment (loss-only, zero inference cost).

    Predict a dense residual offset field from (sr, hr), warp HR toward SR,
    then apply pixel L1 on the warped pair + a small pure-L1 anchor + offset L1.

        L = (1-α) ‖sr − W(hr, off)‖₁ + α ‖sr − hr‖₁ + λ ‖off‖₁

    Offsets are tanh-scaled to ±max_shift px so the aligner cannot collapse
    the task into an arbitrary warp.
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
        """img, offset_px: B,C,H,W / B,2,H,W (offsets in pixels, order dx,dy)."""
        B, _, H, W = img.shape
        ys = torch.linspace(-1.0, 1.0, H, device=img.device, dtype=img.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=img.device, dtype=img.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((gx, gy), dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        sx = 2.0 / max(W - 1, 1)
        sy = 2.0 / max(H - 1, 1)
        scale = torch.empty(B, 2, H, W, device=img.device, dtype=img.dtype)
        scale[:, 0] = sx
        scale[:, 1] = sy
        grid = (base + offset_px * scale).permute(0, 2, 3, 1)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, sr, log_var, hr):
        sr = sr.float()
        hr = hr.float()
        with _autocast_off(sr):
            off = torch.tanh(self.offset_net(torch.cat([sr, hr], dim=1))) * self.max_shift
            hr_aln = self._warp(hr, off)
            l1_aln = (sr - hr_aln).abs().mean()
            l1_pure = (sr - hr).abs().mean()
            l1_off = off.abs().mean()
            total = (1.0 - self.w_pure) * l1_aln + self.w_pure * l1_pure + self.w_off * l1_off
        parts = {
            "l1": l1_pure,
            "l1_aln": l1_aln,
            "off": l1_off,
            "off_mean_px": off.abs().mean(),
            "plain": l1_pure,
            "total": total,
        }
        return total, parts

