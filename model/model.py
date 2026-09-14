"""ModSwinIR v2 (kept): residual bicubic + OCA/GDFN/AMF + FiLM-RAPE + DCN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    DeformAlign,
    RSTB,
    RadialAwarePE,
)


class ModSwinIR(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 2, 2, 2, 2),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        overlap=2,
        mlp_ratio=2.0,
        upscale=2,
        img_range=1.0,
        use_amf=True,
        use_rape=True,
        use_dcn=True,
        film_mid=True,
    ):
        super().__init__()
        self.img_range = img_range
        self.upscale = upscale
        self.window_size = window_size
        self.film_mid = film_mid

        self.register_buffer("rgb_mean", torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.rape = RadialAwarePE(embed_dim, film=True) if use_rape else None
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
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_up = nn.Conv2d(embed_dim, embed_dim * upscale * upscale, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

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
        if self.rape is not None:
            x = self.rape.apply_film(x, self.rape(B, H, W, x.device, x.dtype))
        if self.align is not None:
            x = self.align(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        residual = x
        mid = len(self.body) // 2
        for i, blk in enumerate(self.body):
            x = blk(x)
            if self.rape_mid is not None and i + 1 == mid:
                r = self._radial_grid(H, W, x.device, x.dtype)
                gb = self.rape_mid(r).view(1, H, W, -1).expand(B, -1, -1, -1)
                gamma, beta = gb.chunk(2, dim=-1)
                xb = x.permute(0, 3, 1, 2).contiguous()
                xb = xb * (1.0 + gamma.permute(0, 3, 1, 2)) + beta.permute(0, 3, 1, 2)
                x = xb.permute(0, 2, 3, 1).contiguous()
        x = self.norm_body(x).permute(0, 3, 1, 2).contiguous()
        return self.conv_after_body(x) + residual.permute(0, 3, 1, 2).contiguous()

    def forward(self, lr):
        H, W = lr.shape[-2:]
        x, pad_h, pad_w = self.check_image_size(lr)
        mean = self.rgb_mean * self.img_range
        x_n = (x - mean) * self.img_range
        fea = self.forward_features(x_n)
        res = self.conv_last(self.pixel_shuffle(self.conv_up(fea))) / self.img_range
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        sr = base + res
        if pad_h or pad_w:
            sr = sr[:, :, : H * self.upscale, : W * self.upscale]
        return sr, None


MODEL_SIZES = {
    "tiny": dict(embed_dim=48, depths=(1, 1, 1, 1, 1, 1), num_heads=(3, 3, 3, 3, 3, 3)),
    "base": dict(embed_dim=96, depths=(2, 2, 2, 2, 2, 2), num_heads=(6, 6, 6, 6, 6, 6)),
}


def build_model(upscale=2, size="base", **kwargs):
    cfg = dict(window_size=8, overlap=2, mlp_ratio=2.0, upscale=upscale)
    cfg.update(MODEL_SIZES[size])
    cfg.update(kwargs)
    return ModSwinIR(**cfg)
