"""ModSwinIR v2: math-driven fixes for RealSR V3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    AmplitudeModulatedFourierFFN,
    DeformAlign,
    GatedDconvFFN,
    OverlappingWindowAttention,
    RSTB,
    RadialAwarePE,
)


class LocalKernelHead(nn.Module):
    """LP-KPN: LR-feat → PixelShuffle softmax k×k kernels on bicubic neighborhood."""

    def __init__(self, feat_ch, k=5, upscale=2):
        super().__init__()
        assert k % 2 == 1
        self.k = k
        self.upscale = upscale
        self.pred = nn.Conv2d(feat_ch, k * k * upscale * upscale, 3, 1, 1)
        nn.init.zeros_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)
        center = (k // 2) * k + (k // 2)
        with torch.no_grad():
            for i in range(upscale * upscale):
                self.pred.bias[center * upscale * upscale + i] = 8.0

    def forward(self, feat_lr, img_up):
        """feat_lr: B,Cf,H,W (LR). img_up: B,C,H*s,W*s."""
        B, C, Hs, Ws = img_up.shape
        s, k = self.upscale, self.k
        pad = k // 2
        w = F.pixel_shuffle(self.pred(feat_lr), s)
        w = torch.softmax(w, dim=1)
        img_pad = F.pad(img_up, (pad, pad, pad, pad), mode="reflect")
        patches = F.unfold(img_pad, kernel_size=k, stride=1).view(B, C, k * k, Hs, Ws)
        return (patches * w.unsqueeze(1)).sum(dim=2)


class AmpPhaseHead(nn.Module):
    """
    Image-level amplitude/phase split on residual (inference).

    FFT(res) → amplitude reweighted by exp(tanh(Φ(A))) → phase frozen → IFFT.
    Phase ≈ structure location (misregistration); amplitude ≈ texture energy.
    Zero-init → identity.
    """

    def __init__(self, feat_ch):
        super().__init__()
        self.amp = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feat_ch, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.amp[-1].weight)
        nn.init.zeros_(self.amp[-1].bias)

    def forward(self, feat, img):
        B, C, H, W = img.shape
        with torch.autocast(device_type="cuda" if img.is_cuda else "cpu", enabled=False):
            g = torch.exp(torch.tanh(self.amp(feat.float())))  # B,3,H,W
            x_f = torch.fft.rfft2(img.float(), norm="ortho")
            # broadcast spatial gain onto freq bins via mean per-channel (stable)
            # use spatial mean of g so we don't need FFT-sized maps: global per-channel gain
            g_mean = g.mean(dim=(2, 3), keepdim=True)  # B,3,1,1
            amp = torch.abs(x_f) * g_mean
            phase = torch.angle(x_f)
            out = torch.fft.irfft2(torch.polar(amp, phase), s=(H, W), norm="ortho")
        return out.to(dtype=img.dtype)


class RadialPSFHead(nn.Module):
    """
    Radial PSF residual: out = x + γ(p) ⊙ (x − G_σ * x)  (unsharp / de-PSF).

    γ from features (spatially varying); fixed small Gaussian blur. Zero-init γ → identity.
    """

    def __init__(self, feat_ch, ksize=7, sigma=1.5):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.gain = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feat_ch, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)

    def _gauss_kernel(self, device, dtype):
        k = self.ksize
        coords = torch.arange(k, device=device, dtype=dtype) - k // 2
        g = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g = g / g.sum()
        kernel = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
        return kernel.expand(3, 1, k, k).contiguous()

    def forward(self, feat, img):
        B, C, H, W = img.shape
        with torch.autocast(device_type="cuda" if img.is_cuda else "cpu", enabled=False):
            imgf = img.float()
            ker = self._gauss_kernel(img.device, imgf.dtype)
            blur = F.conv2d(imgf, ker, padding=self.ksize // 2, groups=3)
            hp = imgf - blur
            gamma = 2.0 * torch.tanh(self.gain(feat.float()))  # (-2,2)
            out = imgf + gamma * hp
        return out.to(dtype=img.dtype)


class WienerHead(nn.Module):
    """
    Differentiable radial Wiener/MTF band-gain (inference-time).

    Predict n_bands spatially varying gains g_b = exp(tanh(·)) ∈ (e^{-1}, e^{1}),
    apply them on soft radial frequency masks of the current RGB reconstruction
    (FFT → band-weighted gain → IFFT). Zero-init → identity at step 0.
    """

    def __init__(self, feat_ch, n_bands=4):
        super().__init__()
        self.n_bands = n_bands
        self.to_gain = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feat_ch, n_bands, 3, 1, 1),
        )
        nn.init.zeros_(self.to_gain[-1].weight)
        nn.init.zeros_(self.to_gain[-1].bias)

    def _radial_masks(self, H, W, device, dtype):
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype)
        fx = torch.fft.rfftfreq(W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(gx * gx + gy * gy) / 0.70710678  # ~[0,1]
        centers = torch.linspace(0.0, 1.0, self.n_bands, device=device, dtype=dtype)
        bw = 1.0 / max(self.n_bands - 1, 1)
        masks = []
        for c in centers:
            masks.append(torch.exp(-((r - c) / (bw + 1e-6)) ** 2))
        masks = torch.stack(masks, dim=0)  # n_bands, H, Wf
        masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-8)
        return masks  # n_bands, H, Wf

    def forward(self, feat, img):
        """feat: B,Cf,H,W  img: B,3,H,W → same shape as img (fp32 FFT)."""
        B, C, H, W = img.shape
        device = img.device
        with torch.autocast(device_type="cuda" if img.is_cuda else "cpu", enabled=False):
            gain = torch.exp(torch.tanh(self.to_gain(feat.float())))  # B, nb, H, W
            x_f = torch.fft.rfft2(img.float(), norm="ortho")
            masks = self._radial_masks(H, W, device, torch.float32)  # nb, H, Wf
            out = torch.zeros_like(img.float())
            for b in range(self.n_bands):
                band = torch.fft.irfft2(x_f * masks[b].unsqueeze(0).unsqueeze(0), s=(H, W), norm="ortho")
                out = out + gain[:, b : b + 1] * band
        return out.to(dtype=img.dtype)


class ModSwinIR(nn.Module):
    """
    LR → shallow → RAPE-FiLM → DeformAlign → RSTB×N (OCA+GDFN+AMF, mid FiLM)
       → PixelShuffle → residual on bicubic(LR) → SR (+ optional log-var)

    Reconstruction is residual:  ŷ = Up_bicubic(x) + f_θ(x)
    so the net only has to learn the (real-camera) departure from bilinear
    upsampling — better conditioning than predicting absolute RGB.
    """

    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=48,
        depths=(1, 1, 1, 1, 1, 1),
        num_heads=(3, 3, 3, 3, 3, 3),
        window_size=8,
        overlap=2,
        mlp_ratio=2.0,
        upscale=2,
        img_range=1.0,
        use_amf=True,
        use_rape=True,
        use_dcn=True,
        uncertainty=True,
        residual_recon=True,
        film_mid=True,
        use_kpn=False,
        kpn_size=5,
        use_wiener=False,
        wiener_bands=4,
        use_ampphase=False,
        use_radialpsf=False,
        resi_connection="1conv",
    ):
        super().__init__()
        self.img_range = img_range
        self.upscale = upscale
        self.use_rape = use_rape
        self.use_dcn = use_dcn
        self.uncertainty = uncertainty
        self.residual_recon = residual_recon
        self.film_mid = film_mid
        self.use_kpn = use_kpn
        self.use_wiener = use_wiener
        self.use_ampphase = use_ampphase
        self.use_radialpsf = use_radialpsf
        self.window_size = window_size
        self.embed_dim = embed_dim

        rgb_mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", rgb_mean)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # shallow FiLM from radial PE (identity init)
        self.rape = RadialAwarePE(embed_dim, film=True) if use_rape else None
        # mid-depth FiLM (shared PE MLP is fine — separate γβ for mid)
        self.rape_mid = (
            nn.Sequential(
                nn.Linear(1, max(embed_dim // 2, 8)),
                nn.GELU(),
                nn.Linear(max(embed_dim // 2, 8), embed_dim * 2),
            )
            if (use_rape and film_mid)
            else None
        )
        if self.rape_mid is not None:
            nn.init.zeros_(self.rape_mid[-1].weight)
            nn.init.zeros_(self.rape_mid[-1].bias)

        self.align = DeformAlign(embed_dim) if use_dcn else None

        self.body = nn.ModuleList(
            [
                RSTB(
                    dim=embed_dim,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    overlap=overlap,
                    mlp_ratio=mlp_ratio,
                    depth=depths[i],
                    use_amf=use_amf,
                )
                for i in range(len(depths))
            ]
        )
        self.norm_body = nn.LayerNorm(embed_dim)
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 1)

        self.conv_up = nn.Conv2d(embed_dim, embed_dim * upscale * upscale, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

        # local kernel base (replaces pure bicubic when enabled)
        self.kpn = LocalKernelHead(embed_dim, k=kpn_size, upscale=upscale) if use_kpn else None
        self.wiener = WienerHead(embed_dim, n_bands=wiener_bands) if use_wiener else None
        self.ampphase = AmpPhaseHead(embed_dim) if use_ampphase else None
        self.radial_psf = RadialPSFHead(embed_dim) if use_radialpsf else None

        self.unc_head = (
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim, 1, 3, 1, 1),
            )
            if uncertainty
            else None
        )
        if uncertainty:
            nn.init.zeros_(self.unc_head[-1].weight)
            nn.init.zeros_(self.unc_head[-1].bias)

        self.apply(self._init_weights)
        # re-apply zero inits that _init_weights clobbered
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)
        if uncertainty:
            nn.init.zeros_(self.unc_head[-1].weight)
            nn.init.zeros_(self.unc_head[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        m = self.window_size
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def _radial_grid(self, H, W, device, dtype):
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.sqrt(gx * gx + gy * gy).view(1, H * W, 1)

    def forward_features(self, x):
        B, _, H, W = x.shape
        x = self.conv_first(x)

        r_cache = None
        if self.rape is not None:
            pe = self.rape(B, H, W, x.device, x.dtype)  # B,H,W,2C
            x = self.rape.apply_film(x, pe)
            r_cache = True

        if self.align is not None:
            x = self.align(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        residual = x

        n_blk = len(self.body)
        mid = n_blk // 2
        for i, blk in enumerate(self.body):
            x = blk(x)
            if self.rape_mid is not None and i + 1 == mid:
                r = self._radial_grid(H, W, x.device, x.dtype)
                gb = self.rape_mid(r).view(1, H, W, -1).expand(B, -1, -1, -1)
                gamma, beta = gb.chunk(2, dim=-1)
                gamma = gamma.permute(0, 3, 1, 2).contiguous()
                beta = beta.permute(0, 3, 1, 2).contiguous()
                xb = x.permute(0, 3, 1, 2).contiguous()
                xb = xb * (1.0 + gamma) + beta
                x = xb.permute(0, 2, 3, 1).contiguous()

        x = self.norm_body(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_after_body(x) + residual.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, lr):
        """
        lr: (B, 3, H, W) in [0, 1]
        returns: sr, log_var|None
        """
        H, W = lr.shape[-2:]
        x, pad_h, pad_w = self.check_image_size(lr)
        mean = self.rgb_mean * self.img_range
        x_n = (x - mean) * self.img_range

        fea = self.forward_features(x_n)
        x_up = self.pixel_shuffle(self.conv_up(fea))
        res = self.conv_last(x_up) / self.img_range

        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        if self.kpn is not None:
            # true LP-KPN: space-varying filter of bicubic neighborhood from LR feats
            base = self.kpn(fea, base)

        if self.residual_recon:
            if self.wiener is not None:
                res = self.wiener(x_up, res)
            if self.ampphase is not None:
                res = self.ampphase(x_up, res)
            sr = base + res
        else:
            sr = res + mean

        if self.radial_psf is not None:
            sr = self.radial_psf(x_up, sr)

        log_var = None
        if self.uncertainty and self.unc_head is not None:
            log_var = self.unc_head(x_up)

        if pad_h or pad_w:
            s = self.upscale
            sr = sr[:, :, : H * s, : W * s]
            if log_var is not None:
                log_var = log_var[:, :, : H * s, : W * s]
        return sr, log_var


MODEL_SIZES = {
    "tiny": dict(embed_dim=48, depths=(1, 1, 1, 1, 1, 1), num_heads=(3, 3, 3, 3, 3, 3)),
    "base": dict(embed_dim=96, depths=(2, 2, 2, 2, 2, 2), num_heads=(6, 6, 6, 6, 6, 6)),
    "large": dict(embed_dim=120, depths=(4, 4, 4, 4, 4, 4), num_heads=(6, 6, 6, 6, 6, 6)),
}


def build_model(upscale=2, size="base", **kwargs):
    if size not in MODEL_SIZES:
        raise ValueError(f"unknown size {size}, pick from {list(MODEL_SIZES)}")
    cfg = dict(
        window_size=8,
        overlap=2,
        mlp_ratio=2.0,
        upscale=upscale,
    )
    cfg.update(MODEL_SIZES[size])
    cfg.update(kwargs)
    return ModSwinIR(**cfg)
