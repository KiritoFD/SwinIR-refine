"""Candidate U-Net channel shapes on the "push capacity to low resolution" axis.

All at stride 1 (native_lr=0) unless noted.  FLOPs/params/memory decide how many
runs fit in the wall-clock budget, so measure them before committing.

Step time is estimated from the measured b64 point: 128 samples * 47.23 GFLOP =
6.05 TFLOP/step in 0.587 s -> 10.3 TFLOPS sustained.  Shapes with a narrow level 0
run worse than that (skinny GEMMs), so treat the estimate as a floor.
"""
import sys

sys.path.insert(0, "/home/ds/realsr")

import torch
from torch.utils.flop_counter import FlopCounterMode

from diffusion.unet import build_unet

HR = 128
B = 8
MEASURED_TFLOPS = 10.3  # b64 (1,2,4,4) stride-1, batch 128
MEM_PER_ACT_KB = 0.145  # calibrated: 950k activations -> 137 MB/sample


def measure(base, mult, num_res=2, in_stride=1):
    m = build_unet("S", input_size=HR, in_channels=3, base=base, mult=mult,
                   num_res=num_res, attn_levels=(2, 3),
                   in_stride=in_stride, out_scale=in_stride).eval()
    chans = [max(8, int(round(base * x))) for x in mult]
    top = HR // in_stride
    act = sum(c * (top >> l) ** 2 for l, c in enumerate(chans))  # elements/tensor
    x = torch.randn(B, 3, HR, HR)
    t = torch.zeros(B)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, t)
    return (sum(p.numel() for p in m.parameters()) / 1e6,
            fc.get_total_flops() / B / 1e9,
            act * MEM_PER_ACT_KB / 1000.0,  # MB per sample
            act / 1000.0)                   # kiloelements of activation


# A ladder from the current shape down to constant width.  (1,1,1,1) flattens the
# channel progression entirely, which stops being a U-Net -- so the interesting
# runs are the rungs in between, where the pyramid is still visibly a pyramid.
SHAPES = [
    (64, (1, 2, 4, 4), "1244 b64  (当前)"),
    (80, (1, 2, 4, 4), "1244 b80"),
    (64, (1, 1, 2, 4), "1124 b64"),
    (80, (1, 1, 2, 4), "1124 b80"),
    (96, (1, 1, 2, 4), "1124 b96"),
    (64, (1, 1, 2, 2), "1122 b64"),
    (80, (1, 1, 2, 2), "1122 b80"),
    (96, (1, 1, 2, 2), "1122 b96"),
    (64, (1, 2, 2, 2), "1222 b64"),
    (80, (1, 1, 1, 2), "1112 b80"),
    (96, (1, 1, 1, 1), "1111 b96  (退化)"),
]

print(f"{'shape':16s} {'params':>8s} {'GFLOP':>7s} {'激活':>9s} {'MB/样本':>8s} {'batch':>5s} "
      f"{'s/step':>7s} {'10000步':>7s}   (存/算 = 激活k/GFLOP, 越大越省算力)")
BUDGET = 35_000  # MB, leave headroom on the 48 GB card
for base, mult, note in SHAPES:
    p, f, mb, act_k = measure(base, mult)
    batch = int(BUDGET / mb)
    batch = (batch // 32) * 32  # keep it a round multiple
    step = batch * f / (MEASURED_TFLOPS * 1000)
    hours = 10000 * step / 3600
    ratio = act_k / f
    print(f"{note:16s} {p:7.2f}M {f:7.2f} {act_k:8.0f}k {mb:8.1f} {batch:5d} {step:7.3f} {hours:6.2f}h "
          f" 存/算={ratio:6.1f}", flush=True)
