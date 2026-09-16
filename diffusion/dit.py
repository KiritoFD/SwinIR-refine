"""Modern DiT for RealSR (AdaLN-Zero + RoPE + RMSNorm + SwiGLU + SDPA).

Adapted from G:/GitHub/DiT/src/model — self-contained, no timm.
Input is either latent (C channels) or pixel RGB(+cond concat).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_freqs(dim: int, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """pos (T,) → cos/sin (T, dim). dim must be even."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    freqs = pos.float()[:, None] * inv[None, :]  # (T, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, grid: int, theta: float = 100.0) -> torch.Tensor:
    """Axial 2D RoPE. x: (B, heads, T, hd), T=grid*grid, hd%4==0.

    First half of channels use row (y) position, second half use col (x).
    """
    b, h, t, d = x.shape
    assert t == grid * grid, (t, grid)
    assert d % 4 == 0, d
    device = x.device
    ys = torch.arange(grid, device=device).float()
    xs = torch.arange(grid, device=device).float()
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos_y = gy.reshape(-1)
    pos_x = gx.reshape(-1)
    half = d // 2
    cos_y, sin_y = _rope_freqs(half, pos_y, theta)  # (T, hd/2)
    cos_x, sin_x = _rope_freqs(half, pos_x, theta)
    cos = torch.cat([cos_y, cos_x], dim=-1)[None, None]  # (1,1,T,hd)
    sin = torch.cat([sin_y, sin_x], dim=-1)[None, None]
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, rope: bool = True, rope_theta: float = 100.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        self.rope = rope
        self.rope_theta = rope_theta
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B,T,H,hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope:
            q = apply_rope(q, grid, self.rope_theta)
            k = apply_rope(k, grid, self.rope_theta)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: float = 4.0, qk_norm: bool = True, rope: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm, rope=rope)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, mlp_mult)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, grid: int) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), grid)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DiT(nn.Module):
    """Velocity / eps network. in_channels may include LR condition channels."""

    def __init__(
        self,
        input_size: int,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_mult: float = 4.0,
        qk_norm: bool = True,
        rope: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert input_size % patch_size == 0
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels  # same C; caller slices if cond concat
        self.grid = input_size // patch_size
        self.num_patches = self.grid * self.grid
        self.use_checkpoint = use_checkpoint
        self.depth = depth

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Kept only so older ckpts keep loading; position comes from RoPE, so it
        # stays zero and is never added. That is what lets one ckpt run on any
        # token grid (training crop vs. padded inference tile).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        nn.init.zeros_(self.pos_embed)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_mult, qk_norm=qk_norm, rope=rope)
                for _ in range(depth)
            ]
        )
        self.final = FinalLayer(hidden_size, patch_size, in_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.in_channels
        p = self.patch_size
        g = int(round(math.sqrt(x.shape[1])))
        assert g * g == x.shape[1], f"non-square token grid {x.shape[1]}"
        x = x.reshape(x.shape[0], g, g, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, g * p, g * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) at ANY size divisible by patch_size; t: (B,)."""
        tokens = self.x_embedder(x)
        grid = int(round(math.sqrt(tokens.shape[1])))
        assert grid * grid == tokens.shape[1], (
            f"input {tuple(x.shape[-2:])} -> {tokens.shape[1]} tokens is not a square grid"
        )
        c = self.t_embedder(t)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                tokens = checkpoint(blk, tokens, c, grid, use_reentrant=False)
            else:
                tokens = blk(tokens, c, grid)
        tokens = self.final(tokens, c)
        return self.unpatchify(tokens)


def DiT_S(**kw):
    return DiT(hidden_size=384, depth=12, num_heads=6, **kw)


def DiT_B(**kw):
    return DiT(hidden_size=768, depth=12, num_heads=12, **kw)


def DiT_M(**kw):
    return DiT(hidden_size=512, depth=16, num_heads=8, **kw)


SIZES = {"S": DiT_S, "M": DiT_M, "B": DiT_B}


def build_dit(size: str = "S", **kw) -> DiT:
    return SIZES[size.upper()](**kw)
