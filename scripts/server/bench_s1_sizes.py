"""Params / FLOPs / activations for the stride-1 U-Net series."""
import sys

sys.path.insert(0, "/home/ds/realsr")

import torch
from torch.utils.flop_counter import FlopCounterMode

from diffusion.unet import build_unet

HR = 128
B = 8


def measure(base, mult, num_res=2, attn=(2, 3), in_stride=1):
    m = build_unet("S", input_size=HR, in_channels=3, base=base, mult=mult,
                   num_res=num_res, attn_levels=attn,
                   in_stride=in_stride, out_scale=in_stride).eval()
    chans = [max(8, int(round(base * x))) for x in mult]
    top = HR // in_stride
    act = sum(c * (top >> l) ** 2 for l, c in enumerate(chans)) / 1e3
    x = torch.randn(B, 3, HR, HR)
    t = torch.zeros(B)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, t)
    return (sum(p.numel() for p in m.parameters()) / 1e6,
            fc.get_total_flops() / B / 1e9, act)


SHAPES = [
    (32, (1, 2, 4, 4), "s1 b32 1244"),
    (40, (1, 2, 4, 4), "s1 b40 1244"),
    (48, (1, 2, 4, 4), "s1 b48 1244"),
    (64, (1, 2, 4, 4), "s1 b64 1244"),
    (32, (1, 1, 2, 8), "s1 b32 1128"),
    (48, (1, 1, 2, 8), "s1 b48 1128"),
    (64, (1, 1, 2, 8), "s1 b64 1128"),
]

print(f"{'shape':14s} {'params':>8s} {'GFLOP':>8s} {'act':>8s} {'MB/sample':>10s} {'batch@35G':>10s}")
for base, mult, note in SHAPES:
    p, f, a = measure(base, mult)
    mb = a * 0.145          # calibrated: b32/1244/s1 measured 137 MB/sample at act=950k
    print(f"{note:14s} {p:7.2f}M {f:8.2f} {a:7.1f}k {mb:10.1f} {35000 / mb:10.0f}", flush=True)
