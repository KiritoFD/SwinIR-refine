"""Real GPU MFU for our backbones, measured the normal way:
 - torch.utils.flop_counter.FlopCounterMode counts fwd+bwd matmul/conv/SDPA FLOPs
   (attention included), on a bf16-autocast training step (fwd + backward + opt.step).
 - a timed fwd+bwd loop gives sec/iter.
 - MFU = total_flops_per_iter / (peak * sec_per_iter);  peak = RTX 4090 bf16 dense.
Run: python docs/bench_mfu2.py
"""
import time, argparse
import torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

p = argparse.ArgumentParser()
p.add_argument("--peak", type=float, default=165.2e12, help="4090 bf16 dense TFLOPS")
p.add_argument("--iters", type=int, default=20)
p.add_argument("--warmup", type=int, default=8)
a = p.parse_args()
dev = "cuda"


def build_models():
    from diffusion.unet import build_unet
    from diffusion.dit import build_dit
    out = {}
    out["PixelUNet b64"]  = (build_unet("S", input_size=128, in_channels=3, out_channels=3, base=64,
                                         mult=(1, 2, 4, 4), num_res=2, attn_levels=(2, 3),
                                         in_stride=1, out_scale=1).to(dev), 3, 64)
    out["PixelUNet b128"] = (build_unet("S", input_size=128, in_channels=3, out_channels=3, base=128,
                                         mult=(1, 2, 4, 4), num_res=2, attn_levels=(2, 3),
                                         in_stride=1, out_scale=1).to(dev), 3, 32)
    out["DiT-S (HR128,p2)"] = (build_dit("S", input_size=128, patch_size=2, in_channels=6).to(dev), 6, 16)
    return out


def run_step(model, x, t, opt):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, t)
        loss = out.float().pow(2).mean()   # generic surrogate for a fwd+bwd step's flops
    loss.backward()
    opt.step()


def measure(model, cin, batch):
    x = torch.randn(batch, cin, 128, 128, device=dev)
    t = torch.zeros(batch, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # count fwd+bwd FLOPs on one step
    with FlopCounterMode(display=False) as fc:
        run_step(model, x, t, opt)
    flops = fc.get_total_flops()          # includes backward matmuls (fwd+bwd)
    for _ in range(a.warmup):
        run_step(model, x, t, opt)
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(a.iters):
        run_step(model, x, t, opt)
    torch.cuda.synchronize()
    sec = (time.perf_counter() - s) / a.iters
    achieved = flops / sec
    mfu = 100 * achieved / a.peak
    return flops / 1e9, sec, achieved / 1e12, mfu, batch


print(f"{'backbone':16s} {'bs':>3s} {'fwd+bwd GFLOP/img':>18s} {'s/iter':>8s} {'achieved TFLOPS':>16s} {'MFU %':>7s}")
for name, (model, cin, batch) in build_models().items():
    gf, sec, tf, mfu, b = measure(model, cin, batch)
    per_img = gf / b
    print(f"{name:16s} {b:>3d} {per_img:>18.1f} {sec:>8.3f} {tf:>16.1f} {mfu:>7.1f}")
