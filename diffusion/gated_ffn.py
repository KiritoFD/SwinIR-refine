"""Spatial-gated feed-forward block (continuous / implicit MoE).

Two ideas in one block:

1. **Multiplicative gating instead of stacked channels.**  Two linear projections,
   one of them passed through SiLU and multiplied element-wise into the other.
   That product is a second-order term in the input, so the block expresses
   interactions a plain ReLU/GELU stack cannot, for far fewer FLOPs than widening
   the channel count to get the same hypothesis space.  This is the SwiGLU /
   NAFNet-SimpleGate construction, and it is also what makes it an *implicit*
   mixture of experts: the gate is generated from the features themselves, so
   every spatial location gets its own continuous blend of the basis functions in
   the expansion, with no discrete routing, no load balancing and no patch-level
   expert switching -- which is what keeps the output spatially continuous.

2. **A depthwise 3x3 in the expanded space.**  Without it the two projections are
   per-pixel, and the gate can destroy pixel phase.  The depthwise conv is the
   spatial anchor.

Tensor Core alignment
---------------------
Channel counts are forced to multiples of `align` (default 128).  A GEMM whose
K/N is not a multiple of the tile size pays for silent padding and a downgraded
kernel; at 128 channels wide and 100+ blocks deep that shows up as a large slice
of the runtime.  Misaligned configs crash here rather than train slowly.

Note on the Linear formulation
------------------------------
The reference implementation is written with `nn.Linear` on a [B, H, W, C]
layout.  A 1x1 convolution on [B, C, H, W] applies exactly the same per-pixel
linear map, and it avoids two permutes (and their copies) per call -- which
matters because this block sits inside every residual block.

Cost, for input width C and expansion r: proj_in is C -> 2rC and proj_out is
rC -> C, both 1x1, so about (2r + r) * C^2 = 3r C^2 MACs per position.  At r=3
that is 9 C^2, roughly half of one 3x3 conv's 18 C^2 -- so the block adds
~25% to a two-conv residual block's FLOPs while adding considerably more than
25% of its parameters.

The output projection is zero-initialised, so the block starts as an identity
and never breaks the "a fresh net reproduces its input" property the reg arms
depend on.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGatedFFN(nn.Module):
    """norm-free, residual, 128-aligned gated FFN with a depthwise spatial anchor."""

    def __init__(self, dim: int, expansion_ratio: float = 2.66, align: int = 128):
        super().__init__()
        assert dim % align == 0, (
            f"input dim {dim} fails Tensor Core {align}-alignment; "
            f"next multiples are {(dim // align) * align} and {(dim // align + 1) * align}"
        )
        # round to the nearest aligned width rather than truncating: the ratio the
        # caller asked for is a target, the alignment is a hard constraint
        hidden_dim = int(round(dim * expansion_ratio / align)) * align
        assert hidden_dim % align == 0, f"hidden_dim {hidden_dim} fails {align}-alignment"
        assert hidden_dim >= align, f"hidden_dim {hidden_dim} too small for dim {dim}"

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expansion = hidden_dim / dim

        # -> 2 * hidden so it can be split into the gate and the value
        self.proj_in = nn.Conv2d(dim, 2 * hidden_dim, 1)
        self.dwconv = nn.Conv2d(
            2 * hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1, groups=2 * hidden_dim
        )
        self.proj_out = nn.Conv2d(hidden_dim, dim, 1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden={self.hidden_dim} (x{self.expansion:.2f})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(self.proj_in(x))
        gate, value = h.chunk(2, dim=1)
        return x + self.proj_out(F.silu(gate) * value)
