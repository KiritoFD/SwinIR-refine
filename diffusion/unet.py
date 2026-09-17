"""U-Net backbone for pixel-space SR, drop-in compatible with the DiT interface.

Why this exists
---------------
The DiT-S keeps d=384 tokens at full HR through all 12 layers, so at HR 128
(n = 64^2 = 4096 tokens) the attention term n^2 dominates:

    per sample per layer:  attention 25.8 GFLOP  (64%)  |  qkv+proj 4.8  |  MLP 9.7
    per training step:     40.6 TFLOP / 0.40 s = 101 TFLOPS = 61% of the 4090's
                           bf16 peak -> genuinely compute-bound

and 64% of that compute is spent on full-resolution n^2 attention, which for a
largely local task like SR is a poor use of FLOPs.  It is also why raising the
batch stopped helping.

A U-Net attacks both:
  * activation memory is ~2*H^2*C and independent of depth, instead of
    L*n*d  (measured: 737 MB/sample for DiT-S at HR 128)
  * convolutions lower to implicit GEMM with K = C_in*k^2 (= 864 here), which
    suits tensor cores better than head_dim = 64 attention
  * the full-resolution path stays narrow (base channels only), so the bulk of
    the parameters and FLOPs sit at 1/2, 1/4, 1/8 resolution

Interface is deliberately identical to DiT so train_pixel / train_latent /
eval_official / flow.py need no changes beyond picking the class:

    forward(x: (B, C, H, W), t: (B,) in [0, 1000]) -> (B, C, H, W)

Note on t: flow.py passes ``t * 1000``.  We rescale back to [0, 1] inside the
embedder so the sinusoidal frequencies are meaningful.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c), c)


class TimestepEmbedding(nn.Module):
    """sinusoidal(t) -> 2-layer MLP.  Input t is in [0, 1000]; rescaled to [0,1]."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = (t.float() / 1000.0)[:, None] * freqs[None] * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv -> (+temb) -> GroupNorm -> SiLU -> Conv -> + skip.

    The second conv is zero-initialised (as in DDPM's U-Net), so every block
    starts as an identity and a freshly built net reproduces its input.  That is
    what makes the reg arms start exactly at the bicubic floor.
    """

    def __init__(self, cin: int, cout: int, t_dim: int):
        super().__init__()
        self.norm1 = _gn(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(t_dim, cout)
        self.norm2 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.temb.weight)
        nn.init.zeros_(self.temb.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Single-scale self-attention with a zero-init output projection.

    Only used at the two coarsest levels, where the token count is 1024 and 256,
    so the n^2 term is negligible.
    """

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = _gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = (t.transpose(-1, -2) for t in (qkv[:, 0], qkv[:, 1], qkv[:, 2]))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """Encoder/decoder with skip connections; full resolution carries only `base` channels."""

    def __init__(
        self,
        input_size: int = 128,
        in_channels: int = 3,
        out_channels: int | None = None,
        base: int = 64,
        mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res: int = 2,
        attn_levels: tuple[int, ...] = (2, 3),
        t_dim: int = 256,
        use_checkpoint: bool = False,
        in_stride: int = 1,
        out_scale: int = 1,
        **_unused,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        # flow.py concats the condition as extra input channels and then slices
        # the output back to x0.shape[1]; mirror the DiT and emit in_channels.
        self.out_channels = out_channels or in_channels
        self.use_checkpoint = use_checkpoint
        self.base = base
        self.mult = tuple(mult)
        self.in_stride = int(in_stride)
        self.out_scale = int(out_scale)

        chans = [base * m for m in self.mult]
        self.chans = chans
        L = len(chans)

        # in_stride / out_scale move the whole encoder+decoder down to LR scale
        # and lift it back with a pixel shuffle.  Every skip then joins two
        # maps at the SAME resolution, and the only work left at full HR is one
        # stem conv and one shuffle conv.  This is only valid when the input is
        # band-limited (i.e. the bicubic upsample, as in the reg arms): for flow
        # the ODE state carries white noise at HR and downsampling it would
        # alias away information the velocity field has to reproduce, so the
        # flow arms must keep in_stride = out_scale = 1.
        self.align = self.in_stride * (2 ** (L - 1)) * self.out_scale

        self.t_embed = TimestepEmbedding(t_dim)
        self.stem = nn.Conv2d(in_channels, base, 3, stride=self.in_stride, padding=1)

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = base
        for i, c in enumerate(chans):
            blocks = nn.ModuleList()
            for j in range(num_res):
                blocks.append(ResBlock(prev if j == 0 else c, c, t_dim))
            if i in attn_levels:
                blocks.append(AttnBlock(c))
            self.enc.append(blocks)
            self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < L - 1 else None)
            prev = c

        self.mid = nn.ModuleList(
            [ResBlock(chans[-1], chans[-1], t_dim), AttnBlock(chans[-1]), ResBlock(chans[-1], chans[-1], t_dim)]
        )

        # up[i]: c_{i+1} -> c_i (resolution x2).  dec[i] consumes cat(x, skip) = 2*c_i.
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(chans[i + 1], chans[i], 3, padding=1))
                for i in range(L - 1)
            ]
        )
        self.dec = nn.ModuleList()
        for i in range(L):
            blocks = nn.ModuleList()
            for j in range(num_res):
                blocks.append(ResBlock(2 * chans[i] if j == 0 else chans[i], chans[i], t_dim))
            if i in attn_levels:
                blocks.append(AttnBlock(chans[i]))
            self.dec.append(blocks)

        self.out = nn.Sequential(
            _gn(base), nn.SiLU(),
            nn.Conv2d(base, self.out_channels * self.out_scale**2, 3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _run_blocks(self, blocks, x, temb):
        for blk in blocks:
            if isinstance(blk, ResBlock):
                if self.use_checkpoint and self.training:
                    x = checkpoint(blk, x, temb, use_reentrant=False)
                else:
                    x = blk(x, temb)
            else:
                x = blk(x)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.t_embed(t)

        h = self.stem(x)
        skips = []
        for i, blocks in enumerate(self.enc):
            h = self._run_blocks(blocks, h, temb)
            skips.append(h)
            if self.down[i] is not None:
                h = self.down[i](h)

        h = self._run_blocks(self.mid, h, temb)

        for i in reversed(range(len(self.chans))):
            if i < len(self.chans) - 1:
                h = self.up[i](h)
            h = torch.cat([h, skips[i]], dim=1)
            h = self._run_blocks(self.dec[i], h, temb)

        h = self.out(h)
        if self.out_scale > 1:
            h = F.pixel_shuffle(h, self.out_scale)
        return h


def build_unet(size: str = "S", **kw) -> UNet:
    """`size` scales `base` only; the preset names mirror the DiT ones so the
    same --size flag can drive either backbone."""
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    # `base=0` is the "unset" sentinel used by train_pixel's --base, so treat it
    # (and anything falsy) as "take it from the preset".  Passing 0 through gave
    # chans = [0,0,0,0] and GroupNorm(0, 0) blew up with a modulo-by-zero.
    if not kw.get("base"):
        kw["base"] = preset.get(str(size).upper(), 64)
    kw.pop("patch_size", None)  # DiT-only argument
    return UNet(**kw)
