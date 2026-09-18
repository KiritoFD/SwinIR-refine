"""Shape/param/FLOP check for the width-scaled + gated-FFN U-Nets.

Also asserts the two properties the design depends on:
  * a fresh net is the identity (so reg arms still start at the bicubic floor)
  * misaligned channel widths crash loudly instead of training slowly
"""
import sys

sys.path.insert(0, "/home/ds/realsr")

import torch
from torch.utils.flop_counter import FlopCounterMode

from diffusion.unet import build_unet

HR = 128
B = 8
MEASURED_TFLOPS = 10.3
MEM_PER_ACT_KB = 0.145
BUDGET_MB = 35_000


def measure(base, mult, ffn, ffn_ratio=2.66):
    m = build_unet("S", input_size=HR, in_channels=3, base=base, mult=mult,
                   num_res=2, attn_levels=(2, 3), in_stride=1, out_scale=1,
                   ffn=ffn, ffn_ratio=ffn_ratio).eval()
    chans = [max(8, int(round(base * x))) for x in mult]
    act = sum(c * (HR >> l) ** 2 for l, c in enumerate(chans))
    x = torch.randn(B, 3, HR, HR)
    t = torch.zeros(B)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, t)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    gflop = fc.get_total_flops() / B / 1e9
    mb = act * MEM_PER_ACT_KB / 1000.0
    return m, params, gflop, mb, chans


CONFIGS = [
    (64, (1, 2, 4, 4), False, "1244 b64    (当前)"),
    (128, (1, 2, 4, 4), False, "1244 b128   (纯加宽)"),
    (128, (1, 2, 4, 4), True, "1244 b128 + FFN"),
]

print(f"{'config':22s} {'params':>8s} {'GFLOP':>8s} {'MB/样本':>8s} {'batch':>6s} "
      f"{'s/step':>7s} {'10000步':>7s}")
for base, mult, ffn, note in CONFIGS:
    m, p, f, mb, chans = measure(base, mult, ffn)
    batch = (int(BUDGET_MB / mb) // 16) * 16
    step = batch * f / (MEASURED_TFLOPS * 1000)
    hours = 10000 * step / 3600
    print(f"{note:22s} {p:7.2f}M {f:8.2f} {mb:8.1f} {batch:6d} {step:7.3f} {hours:6.2f}h  chans={chans}",
          flush=True)

# --- properties -----------------------------------------------------------
print("\n--- properties ---")
m, *_ = measure(128, (1, 2, 4, 4), True)
x = torch.randn(2, 3, HR, HR)
with torch.no_grad():
    y = m(x, torch.zeros(2))
print("fresh gated net is identity      :", bool(torch.allclose(x, y, atol=1e-5)),
      f"(max |y-x| = {(y - x).abs().max():.2e})")

try:
    build_unet("S", input_size=HR, in_channels=3, base=64, mult=(1, 2, 4, 4),
               in_stride=1, out_scale=1, ffn=True)
    print("misaligned base 64 + ffn         : NO CRASH  <-- BAD")
except AssertionError as e:
    print("misaligned base 64 + ffn crashes :", str(e)[:90])

ffn = [m_ for m_ in m.modules() if type(m_).__name__ == "SpatialGatedFFN"]
print(f"gated FFN blocks                 : {len(ffn)}")
print(f"expansion                        : {ffn[0].dim} -> {ffn[0].hidden_dim} "
      f"(x{ffn[0].expansion:.2f})")
print(f"hidden alignment % 128           : {ffn[0].hidden_dim % 128}")
