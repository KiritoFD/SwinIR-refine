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


# --------------------------------------------------------------------- SRResNet


class SRResNetBlock(nn.Module):
    """SRResNet residual block: conv-BN-ReLU-conv-BN (BN is its signature vs EDSR)."""

    def __init__(self, nf, k=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, k, padding=k // 2), nn.BatchNorm2d(nf), nn.PReLU(),
            nn.Conv2d(nf, nf, k, padding=k // 2), nn.BatchNorm2d(nf),
        )

    def forward(self, x):
        return x + self.body(x)


class SRResNet(nn.Module):
    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_res=16,
                 scale=2, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = nn.Conv2d(self.in_channels, nf, 9, padding=4)
        self.body = nn.Sequential(*[SRResNetBlock(nf) for _ in range(n_res)])
        self.body_tail = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf))
        self.tail = nn.Conv2d(nf, self.out_channels, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        f = self.body(self.head(x))
        f = self.body_tail(f) + self.head(x)
        return self.tail(f)  # pre-upsampling residual form (see module docstring)


def build_srresnet(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return SRResNet(**kw)


# ---------------------------------------------------------------------- RRDB


class DenseBlock(nn.Module):
    """ESRGAN/Real-ESRGAN dense block: 5 convs, each seeing all previous features,
    residual-scaled by beta."""

    def __init__(self, nf, gc=32, k=3, beta=0.2):
        super().__init__()
        self.beta = beta
        self.convs = nn.ModuleList([
            nn.Conv2d(nf + i * gc, gc, k, padding=k // 2) for i in range(5)
        ])
        self.lff = nn.Conv2d(nf + 5 * gc, nf, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feats = [x]
        for conv in self.convs:
            feats.append(self.act(conv(torch.cat(feats, 1))))
        return self.lff(torch.cat(feats, 1)) * self.beta + x


class RRDB(nn.Module):
    """Residual-in-residual dense block: 3 DenseBlocks, residual-scaled by beta."""

    def __init__(self, nf, gc=32, beta=0.2, n_dense=3):
        super().__init__()
        self.beta = beta
        self.blocks = nn.Sequential(*[DenseBlock(nf, gc) for _ in range(n_dense)])

    def forward(self, x):
        return self.blocks(x) * self.beta + x


class RRDBNet(nn.Module):
    """RRDB backbone (ESRGAN lineage), pre-upsampling residual variant for this
    harness (the standard sub-pixel upsampler would double-upsample the HR input)."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_basic=3,
                 n_dense=3, gc=32, scale=2, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = nn.Conv2d(self.in_channels, nf, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(nf, gc, 0.2, n_dense) for _ in range(n_basic)])
        self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)
        self.tail = nn.Conv2d(nf, self.out_channels, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        f = self.head(x)
        f = self.body_tail(self.body(f)) + f
        return self.tail(f)


def build_rrdb(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return RRDBNet(**kw)
