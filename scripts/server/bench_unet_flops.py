"""Params and FLOPs per sample for candidate U-Net shapes.

The 4090 is capped on effective compute (measured 27 TFLOPS, 17% MFU, 450 W
power limit) while memory sits idle (45% of VRAM, 5% of bandwidth).  So the
design goal is fewest FLOPs per sample at a given parameter count -- memory is
free, compute is not.

Two levers, neither of which changes the parameter count:
  in_stride/out_scale  4 instead of 2  -> the whole net runs at HR/4
  mult                 1,1,2,8 instead of 1,2,4,4 -> capacity moves down
"""
import sys

sys.path.insert(0, "/home/ds/realsr")

import torch
from torch.utils.flop_counter import FlopCounterMode

from diffusion.unet import build_unet

HR = 128
B = 8


def measure(base, mult, num_res, in_stride, out_scale, attn=(2, 3)):
    m = build_unet("S", input_size=HR, in_channels=3, base=base, mult=mult,
                   num_res=num_res, attn_levels=attn,
                   in_stride=in_stride, out_scale=out_scale).eval()
    x = torch.randn(B, 3, HR, HR)
    t = torch.zeros(B)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, t)
    flops = fc.get_total_flops() / B
    params = sum(p.numel() for p in m.parameters()) / 1e6
    return params, flops / 1e9


CONFIGS = [
    ("current   base64 (1,2,4,4) s2", 64, (1, 2, 4, 4), 2, 2, 2),
    ("stride4   base64 (1,2,4,4) s4", 64, (1, 2, 4, 4), 2, 4, 4),
    ("lowres    base64 (1,1,2,8) s2", 64, (1, 1, 2, 8), 2, 2, 2),
    ("both      base64 (1,1,2,8) s4", 64, (1, 1, 2, 8), 2, 4, 4),
    ("xlowres   base64 (1,1,1,16) s4", 64, (1, 1, 1, 16), 2, 4, 4),
    ("both+nr3  base64 (1,1,2,8) s4 r3", 64, (1, 1, 2, 8), 3, 4, 4),
    ("wide+lowres base96 (1,1,2,8) s4", 96, (1, 1, 2, 8), 2, 4, 4),
]

print(f"{'config':34s} {'params':>8s} {'GFLOP/sample':>13s} {'vs current':>11s}")
base_flops = None
for name, b_, mult, nr, si, so in CONFIGS:
    p, f = measure(b_, mult, nr, si, so)
    if base_flops is None:
        base_flops = f
    print(f"{name:34s} {p:8.2f}M {f:13.2f} {base_flops/f:10.2f}x", flush=True)
