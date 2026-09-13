"""Mod-SwinIR building blocks: OCA, GDFN, AMF, RAPE, DeformAlign."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


def window_partition(x, window_size):
    """(B, H, W, C) -> (B*nW, wh, ww, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(B*nW, wh, ww, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class OverlappingWindowAttention(nn.Module):
    """OCA (HAT-style): Q from MxM windows, K/V from (M+2O)x(M+2O) expanded windows."""

    def __init__(self, dim, window_size=8, overlap=2, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.overlap = overlap
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        kv_size = (window_size + 2 * overlap) ** 2
        q_size = window_size**2

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # full Q x Kv relative bias (stable, no index bookkeeping)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(q_size, kv_size, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        M = self.window_size
        O = self.overlap
        assert H % M == 0 and W % M == 0, f"H,W must divide window, got {H},{W},{M}"

        nWh, nWw = H // M, W // M
        nW = nWh * nWw

        # Q: non-overlapping windows
        x_q = window_partition(x, M)  # (B*nW, M, M, C)
        q = self.q(x_q).view(-1, M * M, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # (B*nW, heads, Q, head_dim)

        # K/V: overlapping expanded windows via unfold on padded map
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        x_pad = F.pad(x_bchw, (O, O, O, O), mode="reflect")
        kx = M + 2 * O
        kv_unf = F.unfold(x_pad, kernel_size=kx, stride=M)  # B, C*kx*kx, nW
        kv_unf = kv_unf.view(B, C, kx, kx, nW).permute(0, 4, 2, 3, 1).contiguous()
        kv_unf = kv_unf.view(B * nW, kx * kx, C)
        kv = self.kv(kv_unf).view(-1, kx * kx, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  # 2, B*nW, heads, Kv, head_dim
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B*nW, heads, Q, Kv

        bias = self.relative_position_bias_table.permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(-1, M * M, C)
        out = self.proj_drop(self.proj(out))
        out = out.view(-1, M, M, C)
        return window_reverse(out, M, H, W)


class GatedDconvFFN(nn.Module):
    """GDFN (NAFNet): X1 ⊙ DWConv(X2), no GELU after gating conv."""

    def __init__(self, dim, expansion=2.0, bias=True):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        """x: (B, C, H, W)"""
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        return self.project_out(x1 * self.dwconv(x2))


class AmplitudeModulatedFourierFFN(nn.Module):
    """
    AMF v2 — residual spectral gain.

    v1 used A' = A * sigmoid(Φ(A)) ∈ (0, A], so it could only *suppress*
    frequencies. Texture enhancement needs a map that can both boost and cut:
        A' = A * (1 + tanh(Φ(A))) ∈ (0, 2A]
    Phase is frozen (misregistration ≈ phase error); amplitude carries contrast
    / high-frequency energy we want to reweight.
    """

    def __init__(self, dim, bias=True, gain=1.0):
        super().__init__()
        self.gain = gain
        self.amp_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=bias),
        )
        # start near identity: Φ ≈ 0 => A' ≈ A
        nn.init.zeros_(self.amp_conv[-1].weight)
        nn.init.zeros_(self.amp_conv[-1].bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        x32 = x.float()
        x_f = torch.fft.rfft2(x32, norm="ortho")
        amp = torch.abs(x_f)
        phase = torch.angle(x_f)
        amp_mod = amp * (1.0 + self.gain * torch.tanh(self.amp_conv(amp)))
        x_mod = torch.polar(amp_mod, phase)
        out = torch.fft.irfft2(x_mod, s=(h, w), norm="ortho")
        return out.to(dtype=x.dtype)


class RadialAwarePE(nn.Module):
    """
    RAPE: radial distance r=||(x,y)|| from optical center → FiLM parameters.

    Lens PSF is space-variant (stronger blur / CA toward corners). Instead of a
    single additive PE that the trunk can wash out, map r → (γ, β) and FiLM-modulate
    features at multiple depths:
        x ← x ⊙ (1 + γ(r)) + β(r)
    γ, β are channel-wise; r is shared per spatial location.
    """

    def __init__(self, dim, hidden=None, film=True):
        super().__init__()
        hidden = hidden or max(dim // 2, 8)
        self.film = film
        out_dim = dim * 2 if film else dim
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        # identity FiLM / zero PE at init
        if film:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        self._cache = None

    def _radius(self, H, W, device, dtype):
        key = (H, W)
        if self._cache is None or self._cache[0] != key:
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            r = torch.sqrt(gx * gx + gy * gy).view(1, H * W, 1)
            self._cache = (key, r)
        return self._cache[1].to(device=device, dtype=dtype)

    def forward(self, B, H, W, device, dtype):
        """Returns PE/FiLM params as (B, H, W, C) or (B, H, W, 2C)."""
        r = self._radius(H, W, device, dtype)
        out = self.mlp(r).view(1, H, W, -1)
        return out.expand(B, -1, -1, -1).contiguous()

    def apply_film(self, x_bchw, pe_bhw2c):
        """x: BCHW; pe from forward (film=True): BHW2C → FiLM-modulated BCHW."""
        gamma, beta = pe_bhw2c.chunk(2, dim=-1)
        gamma = gamma.permute(0, 3, 1, 2).contiguous()
        beta = beta.permute(0, 3, 1, 2).contiguous()
        return x_bchw * (1.0 + gamma) + beta


class DeformAlign(nn.Module):
    """Lightweight deformable alignment (torchvision DCN) before deep trunk."""

    def __init__(self, dim, kernel_size=3, num_points=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_points = num_points
        offset_channels = 2 * num_points * kernel_size * kernel_size // 9  # for k=3 -> 18? keep simple
        # standard: offset has 2 * kH * kW channels for one group
        self.offset_mask = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, 2 * kernel_size * kernel_size + kernel_size * kernel_size, 3, 1, 1),
        )
        nn.init.zeros_(self.offset_mask[-1].weight)
        nn.init.zeros_(self.offset_mask[-1].bias)
        self.weight = nn.Parameter(torch.eye(dim).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        k = self.kernel_size
        om = self.offset_mask(x)
        offset = om[:, : 2 * k * k]
        mask = torch.sigmoid(om[:, 2 * k * k :])
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=1,
            padding=k // 2,
            mask=mask,
        )


class RSTB(nn.Module):
    """Residual Swin Transformer Block: [Norm-OCA-Add] [Norm-GDFN-Add] [Norm-AMF-Add] xN + conv residual."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        overlap=2,
        mlp_ratio=2.0,
        depth=1,
        use_amf=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ModTransformerBlock(dim, num_heads, window_size, overlap, mlp_ratio, use_amf) for _ in range(depth)]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.dim = dim

    def forward(self, x):
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        residual = x
        for blk in self.blocks:
            x = blk(x)
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        x_bchw = self.conv(x_bchw)
        x = x_bchw.permute(0, 2, 3, 1).contiguous()
        return x + residual


class ModTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, overlap, mlp_ratio, use_amf=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OverlappingWindowAttention(dim, window_size, overlap, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedDconvFFN(dim, mlp_ratio)
        self.use_amf = use_amf
        if use_amf:
            self.norm3 = nn.LayerNorm(dim)
            self.amf = AmplitudeModulatedFourierFFN(dim)

    def forward(self, x):
        # OCA (channels-last)
        x = x + self.attn(self.norm1(x))
        # GDFN (channels-first)
        x_b = x.permute(0, 3, 1, 2).contiguous()
        x_b = x_b + self.ffn(self.norm2(x).permute(0, 3, 1, 2).contiguous())
        x = x_b.permute(0, 2, 3, 1).contiguous()
        # AMF
        if self.use_amf:
            x_b = x.permute(0, 3, 1, 2).contiguous()
            x_b = x_b + self.amf(self.norm3(x).permute(0, 3, 1, 2).contiguous())
            x = x_b.permute(0, 2, 3, 1).contiguous()
        return x
