"""Params vs activations vs FLOPs for U-Net shapes, to expose the trade-off.

VRAM is dominated by activations, and activations per FLOPs are NOT free:

    params      P = 9  * sum_l C_l^2            (conv weights, per conv)
    FLOPs       F = 18 * sum_l C_l^2 * s_l      (s_l = (LR/2^l)^2)
    activations A =      sum_l C_l   * s_l      (the VRAM driver, per tensor)

With F fixed, Cauchy-Schwarz gives A <= sqrt(F/18 * sum_l s_l), with equality
when every C_l is EQUAL.  So:

    A is maximised by constant width      (mult 1,1,1,1 -- the EDSR shape)
    P is maximised by pushing C down       (mult 1,1,2,8 / 1,1,2,16)

The two pull in opposite directions and cannot both be maximal.  `act/18M`
below is activations rescaled to a common parameter count, i.e. how much VRAM
a shape buys per unit of model size; `gflop/18M` is the compute it costs.
"""
import sys

sys.path.insert(0, "/home/ds/realsr")

import torch
from torch.utils.flop_counter import FlopCounterMode

from diffusion.unet import build_unet

HR = 128
B = 8
def measure(base, mult, num_res=2, attn=(2, 3), in_stride=2):
    m = build_unet("S", input_size=HR, in_channels=3, base=base, mult=mult,
                   num_res=num_res, attn_levels=attn,
                   in_stride=in_stride, out_scale=in_stride).eval()
    chans = [max(8, int(round(base * x))) for x in mult]
    # spatial per level: HR/in_stride/2^l, squared
    top = HR // in_stride
    S = [(top >> l) ** 2 for l in range(len(mult))]
    act = sum(c * s for c, s in zip(chans, S)) / 1e3  # kiloelements per tensor
    x = torch.randn(B, 3, HR, HR)
    t = torch.zeros(B)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, t)
    return (sum(p.numel() for p in m.parameters()) / 1e6,
            fc.get_total_flops() / B / 1e9, act)


SHAPES = [
    (32, (1, 2, 4, 4), 2, 1, "s1 b32"),
    (40, (1, 2, 4, 4), 2, 1, "s1 b40"),
    (48, (1, 2, 4, 4), 2, 1, "s1 b48"),
    (64, (1, 2, 4, 4), 2, 1, "s1 b64"),
    (32, (1, 1, 2, 8), 2, 1, "s1 b32 1128"),
    (48, (1, 1, 2, 8), 2, 1, "s1 b48 1128"),
]]

print(f"{'base':>4s} {'mult':>14s} {'params':>8s} {'GFLOP':>7s} {'act':>7s} "
      f"{'gflop/18M':>9s} {'act/18M':>8s}  note")
rows = []
for base, mult, nr, si, note in SHAPES:
    p, f, a = measure(base, mult, nr, in_stride=si)
    rows.append((p, f, a, base, mult, note))
    print(f"{base:>4d} {str(mult):>14s} {p:7.2f}M {f:7.2f} {a:7.1f}k "
          f"{f * 18.68 / p:9.2f} {a * 18.68 / p:8.1f}  {note}", flush=True)

print("\nact/18M  high = more VRAM per unit of model size (good for a memory-rich card)")
print("gflop/18M high = more compute per unit of model size (bad here)")
