"""Classic SR baselines (EDSR / RCAN), re-implemented for THIS harness so we can put
them on the same A-strict protocol as our U-Net instead of quoting paper numbers from a
different protocol.

Faithfulness notes (be explicit — do not overclaim):
  * This harness feeds the **bicubic-upsampled LR at HR resolution** and adds a residual.
    So we implement the **pre-upsampling** variants (EDSR/RCAN paper §4 ablation: input is
    upsampled first, the network refines at HR) rather than the post-upsampling sub-pixel
    head — the sub-pixel head would upsample a second time and is meaningless on HR input.
  * Bodies are standard: EDSR = residual blocks (conv-ReLU-conv, no norm); RCAN = residual
    groups of channel-attention blocks (RCAB) + CA.  Both return `y - x` so the harness's
    `pred = bicubic(x) + model(x)` reconstructs the refined HR image y.
  * So a reader should treat these as "EDSR-/RCAN-style pre-upsampling baselines under an
    identical protocol", not bit-exact copies of the original repos.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(cin, cout, k, bias=True):
    return nn.Conv2d(cin, cout, k, padding=k // 2, bias=bias)


class ResBlock(nn.Module):
    """EDSR residual block (no BN)."""

    def __init__(self, nf, k=3, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(default_conv(nf, nf, k), nn.ReLU(True), default_conv(nf, nf, k))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_res=16,
                 scale=2, res_scale=0.1, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = default_conv(self.in_channels, nf, 3)
        self.body = nn.Sequential(*[ResBlock(nf, 3, res_scale) for _ in range(n_res)])
        self.tail = default_conv(nf, self.out_channels, 3)
        self.align = 1  # pre-upsampling: no internal down/upsample, any HR size works

    def forward(self, x, t=None):
        y = self.tail(self.body(self.head(x)))
        return y  # harness: pred = bicubic(x) + this


class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nf, max(nf // reduction, 4), 1), nn.ReLU(True),
            nn.Conv2d(max(nf // reduction, 4), nf, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv_du(self.avg(x))


class RCAB(nn.Module):
    def __init__(self, nf, k=3, reduction=16, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(nf, nf, k), nn.ReLU(True), default_conv(nf, nf, k), CALayer(nf, reduction)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class RCAN(nn.Module):
    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_res=10,
                 n_groups=5, scale=2, reduction=16, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = default_conv(self.in_channels, nf, 3)
        self.body = nn.Sequential(*[
            nn.Sequential(*[RCAB(nf, 3, reduction) for _ in range(n_res)])
            for _ in range(n_groups)
        ])
        self.body_tail = default_conv(nf, nf, 3)
        self.tail = default_conv(nf, self.out_channels, 3)
        self.align = 1

    def forward(self, x, t=None):
        f = self.head(x)
        f = self.body_tail(self.body(f)) + f
        return self.tail(f)  # harness: pred = bicubic(x) + this


def build_edsr(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return EDSR(**kw)


def build_rcan(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return RCAN(**kw)
