"""Params vs FLOPs per sample for U-Net channel shapes, at stride 2.

in_stride is fixed at 2: HR = 2*LR, so a stride-2 stem lands exactly on the
native LR resolution and is information-preserving.  stride 4 would work at
LR/2 and throw away half of the only information source SR has.

What is left to choose is where the capacity sits.  A weight at level L costs
2 * (LR/2^L)^2 FLOPs per forward conv, so a parameter at the bottleneck is
64x cheaper than one at full LR:

    level 0  64x64 = 4096 positions  ->  8192 FLOPs per weight
    level 1  32x32 = 1024            ->  2048
    level 2  16x16 =  256            ->   512
    level 3   8x8  =   64            ->   128

That is the whole "more memory, less compute" lever for a U-Net: move channels
down.  The open question is how far down before SR quality suffers, since the
high-resolution levels are where detail lives.
"""
import sys

sys.path.insert(0, "/home/ds/realsr")

import torch
from torch.utils.flop_counter import FlopCounterMode

from diffusion.unet import build_unet

HR = 128
B = 8


def measure(base, mult, num_res=2, attn=(2, 3)):
    m = build_unet("S", input_size=HR, in_channels=3, base=base, mult=mult,
                   num_res=num_res, attn_levels=attn, in_stride=2, out_scale=2).eval()
    x = torch.randn(B, 3, HR, HR)
    t = torch.zeros(B)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, t)
    return (sum(p.numel() for p in m.parameters()) / 1e6,
            fc.get_total_flops() / B / 1e9)


SHAPES = [
    (64, (1, 2, 4, 4), 2, "current"),
    (64, (1, 1, 2, 8), 2, "user's 1128"),
    (64, (1, 1, 1, 8), 2, ""),
    (64, (1, 1, 4, 8), 2, ""),
    (64, (1, 2, 2, 8), 2, ""),
    (64, (1, 2, 4, 8), 2, ""),
    (64, (1, 2, 8, 8), 2, ""),
    (64, (1, 1, 2, 16), 2, ""),
    (64, (1, 1, 2, 8), 3, "1128 + num_res3"),
    (96, (1, 1, 2, 8), 2, "wide+1128"),
    (64, (0.5, 2, 4, 4), 2, "narrow L0 only"),
    (64, (0.5, 1, 2, 8), 2, ""),
    (64, (0.5, 2, 4, 8), 2, ""),
    (64, (0.5, 1, 1, 8), 2, ""),
    (64, (0.5, 1, 2, 16), 2, ""),
    (64, (0.25, 1, 2, 8), 2, ""),
]

print(f"{'base':>4s} {'mult':>14s} {'res':>3s} {'params':>9s} {'GFLOP':>8s} "
      f"{'GFLOP/18M':>10s} {'note':>16s}")
rows = []
for base, mult, nr, note in SHAPES:
    p, f = measure(base, mult, nr)
    rows.append((p, f))
    print(f"{base:>4d} {str(mult):>14s} {nr:>3d} {p:8.2f}M {f:8.2f} "
          f"{f * 18.68 / p:10.2f} {note:>16s}", flush=True)

print("\nGFLOP/18M = FLOPs rescaled to a common 18.68M parameters, so shapes can be")
print("compared at equal capacity.  Lower is better: same model size, less compute.")
