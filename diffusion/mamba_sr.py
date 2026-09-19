"""MambaSR: stride-1 VSS (2D selective-scan) backbone for the reg line.

Why this fits what the project measured
---------------------------------------
* stride-1 is the single biggest lever we found (+0.38 dB over the LR-scale
  U-Net) -- this net NEVER downsamples, so nothing has to be pixel-shuffled
  back into existence.
* the U-Net's receptive field is bounded by the pyramid; flattening the
  pyramid lost 0.07 dB, which says the reg task at HR wants LONG-range
  context at full resolution.  Full-res self-attention is O(L^2) (the DiT
  spent 64% of its FLOPs on it); a selective scan is O(L), so global context
  at HR128 is affordable for the first time.
* conv weights are position- and content-INDEPENDENT; the S6 selection
  mechanism (delta_t, B, C all generated from the input) is the "content
  routing" the frequency-router A/B (neutral, -0.02 dB) tried to bolt on
  from the outside -- here it is native to the operator.

Topology (MambaIR-style residual-in-residual, no U-Net pyramid):

    x -> stem 3x3 -> [RG x num_groups] -> 3x3 -> (zero-init head) -> + bicubic

    RG = sum of num_res x VSSBlock, then a zero-init 3x3, plus identity
    VSSBlock = LN -> dwconv 3x3 -> SiLU -> SS2D (4-direction cross scan)
               -> zero-init 1x1 -> + identity

Every residual contributor (block proj, RG tail conv, output head) is
zero-initialised, so a fresh net reproduces its input and the reg arm starts
EXACTLY at the bicubic floor -- the invariant every arm in this repo relies on.

2D scanning: Mamba is a 1D recurrence, images are 2D.  SS2D flattens the map
in 4 orders (row-major, reversed, column-major, reversed), stacks them on the
batch axis so one kernel call covers all four, and mean-fuses the restorations.

Backends (auto-detected, identical interfaces):
  * "mamba_ssm" -- the official fused CUDA kernel (fast path).  Needs
    causal-conv1d-style custom kernels; built on the server from source with
    a conda nvcc.  No code changes where it exists.
  * "torch"     -- a pure-PyTorch S6.  Mamba's recurrence is REAL DIAGONAL:
    h_t = a_t h_{t-1} + b_t with a_t = exp(dt_t * A), A < 0 so a_t in (0,1).
    That has an exact parallel form, h_t = P_t (h_0 + sum_j b_j / P_j),
    P_t = prod a_i, computable with a cumulative sum of log a.  The sequence is
    processed in short segments (renormalisation): dt is clamped to Mamba's own
    init ceiling (0.1), so |log a| <= 1.6/step and a 24-step segment keeps
    exp(cumsum) >= e^-38 -- exactly representable in fp32 INCLUDING the backward
    of the division (the naive long-segment form computes -b/P^2 there, which
    overflows to inf and poisons the gradients with NaN).  The scan itself runs
    fp32 with autocast disabled, because cumsum in bf16 loses 256 steps of
    mantissa.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def mamba_ssm_available() -> bool:
    try:
        from mamba_ssm import Mamba  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------- pure S6


class MambaCore(nn.Module):
    """Pure-PyTorch Mamba block: (B, L, d_model) -> (B, L, d_model).

    Parameter names/shapes mirror mamba_ssm.modules.mamba_simple.Mamba so a
    checkpoint trained with one backend loads into the other.
    """

    # dt is clamped to the same ceiling Mamba's initialisation targets (0.1).
    # This bounds the per-step decay |dt*A| <= 0.1*16 = 1.6, which is what makes
    # the log-space scan below exactly representable in fp32 over a segment:
    # worst-case log-cumsum over 32 steps is -51, i.e. P >= 7e-23 and b/P <= ~1e23,
    # both far inside fp32 range.  Without the ceiling, dt could grow during
    # training until exp(cumsum) underflows and the division b/P overflows.
    DT_MAX = 0.1      # dt ceiling: bounds per-step decay |dt*A| <= 0.1*N

    def __init__(self, d_model: int, d_state: int = 16, expand: float = 2,
                 d_conv: int = 4, chunk: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = int(expand)
        self.d_inner = self.expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16))
        # Segment length.  The parallel scan divides by P (the cumulative decay);
        # its BACKWARD computes -b/P^2, which overflows fp32 once P < ~1e-19.
        # |log a| <= DT_MAX * d_state per step, so a segment of
        # 38 / (DT_MAX * d_state) steps keeps exp(cumsum) >= e^-38 and the
        # backward intermediates <= ~1e33.  (The forward alone would survive
        # longer segments; it is the autograd of the division that sets this.)
        self.chunk = chunk or max(8, min(24, int(38 / (self.DT_MAX * d_state))))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is negative real (S4D-real layout), so a_t = exp(dt*A) lands in (0,1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.act = nn.SiLU()
        self._init_dt()

    def _init_dt(self):
        # Mamba's dt initialisation: softplus(bias) ~ U[dt_min, dt_max], and the
        # weight scaled so dt_proj(x) starts near zero (dt is then bias-driven)
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp_min(1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))     # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        with torch.no_grad():
            self.dt_proj.weight.uniform_(0, self.dt_rank ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)."""
        b, l, _ = x.shape
        xz = self.in_proj(x)
        xh, z = xz.chunk(2, dim=-1)                          # (B,L,D) each
        xh = self.conv1d(xh.transpose(1, 2))[..., :l].transpose(1, 2)
        xh = self.act(xh)

        dbl = self.x_proj(xh)                                # (B,L,rank+2N)
        dt = F.softplus(self.dt_proj(dbl[..., : self.dt_rank]))
        dt = dt.clamp(max=self.DT_MAX)                       # (B,L,D), see DT_MAX
        Bm = dbl[..., self.dt_rank: self.dt_rank + self.d_state]     # (B,L,N)
        C = dbl[..., self.dt_rank + self.d_state:]                   # (B,L,N)

        y = self._scan(xh, dt, Bm, C) + xh * self.D          # (B,L,D)
        return self.out_proj(y * self.act(z))

    def _scan(self, xh: torch.Tensor, dt: torch.Tensor, Bm: torch.Tensor,
              C: torch.Tensor) -> torch.Tensor:
        """Exact segmented scan of h_t = a_t h_{t-1} + (dt_t x_t) B_t.

        Within a segment the parallel form
            h_t = P_t (h_carry + sum_j b_j / P_j),  P_t = prod of a up to t
        is computed with a cumsum of log a.  The segment is short enough that
        P never enters the range where the BACKWARD of the division (b/P^2)
        overflows fp32, so the form stays exact in fp32; the carried h crosses
        segments in absolute terms, which is the renormalisation step.
        """
        amp = torch.amp.autocast(device_type=xh.device.type, enabled=False)
        with amp:
            xh = xh.float()
            dt = dt.float()
            Bm = Bm.float()
            C = C.float()
            A = -torch.exp(self.A_log.float())               # (D,N) < 0

            bsz, L = xh.shape[:2]
            # a and b are rebuilt per segment -- materialising them over the
            # whole L would be (B,L,D,N), i.e. 23 GB for one val image
            h = xh.new_zeros(bsz, self.d_inner, self.d_state)
            ys = []
            for i in range(0, L, self.chunk):
                dt_c = dt[:, i: i + self.chunk]              # (B,Lc,D)
                a = torch.exp(dt_c.unsqueeze(-1) * A)        # (B,Lc,D,N) in (0,1)
                b = (dt_c * xh[:, i: i + self.chunk]).unsqueeze(-1) \
                    * Bm[:, i: i + self.chunk].unsqueeze(2)  # (B,Lc,D,N)
                # -80 is pure underflow insurance (unreachable for in-range dt)
                P = torch.exp(torch.cumsum(
                    torch.log(a.clamp_min(1e-30)), dim=1).clamp(min=-80.0))
                s = torch.cumsum(b / P, dim=1)
                hc = P * (h.unsqueeze(1) + s)                # (B,Lc,D,N)
                ys.append(torch.einsum("bldn,bln->bld", hc, C[:, i: i + self.chunk]))
                h = hc[:, -1]
            return torch.cat(ys, dim=1)


def _naive_scan(xh: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                Bm: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Sequential reference for the smoke test (correct, slow)."""
    a = torch.exp(dt.unsqueeze(-1) * A)
    b = (dt * xh).unsqueeze(-1) * Bm.unsqueeze(2)
    h = xh.new_zeros(xh.shape[0], xh.shape[-1], A.shape[-1])
    ys = []
    for t in range(xh.shape[1]):
        h = a[:, t] * h + b[:, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=1)


# ---------------------------------------------------------------- SS2D


class SS2D(nn.Module):
    """2D selective scan: 4 scan orders, batch-concat, mean fusion."""

    def __init__(self, d_model: int, d_state: int = 16, expand: float = 2,
                 d_conv: int = 4, backend: str = "auto", chunk: int = 0):
        super().__init__()
        if backend == "auto":
            backend = "mamba_ssm" if mamba_ssm_available() else "torch"
        self.backend = backend
        if backend == "mamba_ssm":
            from mamba_ssm import Mamba

            self.core = Mamba(d_model=d_model, d_state=d_state, expand=expand,
                              d_conv=d_conv, use_fast_path=True)
        else:
            # chunk=0 -> MambaCore picks the autograd-safe segment length
            self.core = MambaCore(d_model, d_state, expand, d_conv, chunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)                       # (B, L, C), (h,w) order
        seq_t = x.transpose(2, 3).flatten(2).transpose(1, 2)     # (B, L, C), (w,h) order
        xs = torch.cat([seq, torch.flip(seq, dims=[1]),
                        seq_t, torch.flip(seq_t, dims=[1])], dim=0)
        y = self.core(xs)
        y1, y2, y3, y4 = y.chunk(4, dim=0)
        y1 = y1.transpose(1, 2).reshape(b, c, h, w)
        y2 = torch.flip(y2, dims=[1]).transpose(1, 2).reshape(b, c, h, w)
        y3 = y3.transpose(1, 2).reshape(b, c, w, h).transpose(2, 3)
        y4 = torch.flip(y4, dims=[1]).transpose(1, 2).reshape(b, c, w, h).transpose(2, 3)
        return (y1 + y2 + y3 + y4) * 0.25


# ---------------------------------------------------------------- blocks


class VSSBlock(nn.Module):
    """LN -> dwconv -> SiLU -> SS2D -> zero-init 1x1 -> + identity."""

    def __init__(self, dim: int, d_state: int = 16, expand: float = 2,
                 backend: str = "auto"):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.ss2d = SS2D(dim, d_state, expand, backend=backend)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.act(self.dwconv(h))
        h = self.ss2d(h)
        return x + self.proj(h)


class ResidualGroup(nn.Module):
    """num_res x VSSBlock, then a zero-init 3x3, plus identity."""

    def __init__(self, dim: int, depth: int, d_state: int, expand: float,
                 backend: str, use_checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [VSSBlock(dim, d_state, expand, backend) for _ in range(depth)]
        )
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.use_checkpoint = use_checkpoint
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x + self.conv(x)


class MambaSR(nn.Module):
    """Stride-1 global-residual VSSM; interface-compatible with UNet/DiT.

    forward(x, t) IGNORES t: this is a reg-line backbone (deterministic),
    same contract the reg arms of the U-Net/DiT use (t comes in as zeros).
    """

    def __init__(
        self,
        input_size: int = 128,
        in_channels: int = 3,
        out_channels: int | None = None,
        dim: int = 128,
        num_groups: int = 4,
        num_res: int = 4,
        d_state: int = 16,
        expand: float = 2,
        d_conv: int = 4,
        backend: str = "auto",
        use_checkpoint: bool = False,
        coord_channels: int = 0,
        **_unused,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.coord_channels = int(coord_channels)
        self.out_channels = out_channels or in_channels
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.backend = backend if backend != "auto" else \
            ("mamba_ssm" if mamba_ssm_available() else "torch")

        # coord channels work exactly like the U-Net's: zero-init stem slice, so
        # a coord mamba at step 0 IS the no-coord mamba
        self.stem = nn.Conv2d(in_channels + self.coord_channels, dim, 3, padding=1)
        if self.coord_channels:
            nn.init.zeros_(self.stem.weight[:, in_channels:])

        self.groups = nn.ModuleList(
            [ResidualGroup(dim, num_res, d_state, expand, self.backend, use_checkpoint)
             for _ in range(num_groups)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_conv = nn.Conv2d(dim, self.out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stem(x)
        for g in self.groups:
            h = g(h)
        h = self.out_norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.out_conv(h)


def build_mambasr(size: str = "S", **kw) -> MambaSR:
    """`size` presets dim; --base overrides it (same convention as build_unet)."""
    preset = {"XS": 64, "S": 128, "M": 192, "B": 256}
    if not kw.get("dim"):
        kw["dim"] = preset.get(str(size).upper(), 128)
    kw.pop("patch_size", None)
    return MambaSR(**kw)
