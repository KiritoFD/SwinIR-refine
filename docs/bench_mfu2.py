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


def build_specs():
    from diffusion.unet import build_unet
    from diffusion.dit import build_dit
    specs = []
    for b in (32, 40, 48, 64, 128):
        specs.append((f"UNet b{b} (s1)",
                      lambda b=b: build_unet("S", input_size=128, in_channels=3, out_channels=3, base=b,
                                             mult=(1, 2, 4, 4), num_res=2, attn_levels=(2, 3),
                                             in_stride=1, out_scale=1).to(dev),
                      3, 64 if b <= 64 else 32))
    specs.append(("DiT-S (HR128,p2)",
                  lambda: build_dit("S", input_size=128, patch_size=2, in_channels=6).to(dev), 6, 16))
    try:
        from diffusion.mamba_sr import build_mambasr
        build_mambasr  # availability check only; instantiate lazily below
        specs.append(("Mamba s1 d96",
                      lambda: build_mambasr("S", input_size=128, in_channels=3, dim=96,
                                            num_res=4, backend="cuda", shift=False).to(dev), 3, 16))
    except Exception as e:
        print(f"  (mamba unavailable: {e})")
    return specs


def run_step(model, x, t, opt):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, t)
        loss = out.float().pow(2).mean()   # generic surrogate for a fwd+bwd step's flops
    loss.backward()
    opt.step()


def measure(factory, cin, batch):
    import gc
    model = factory()
    x = torch.randn(batch, cin, 128, 128, device=dev)
    t = torch.zeros(batch, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with FlopCounterMode(display=False) as fc:
        run_step(model, x, t, opt)
    flops = fc.get_total_flops()
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
    del model, opt, x, t; gc.collect(); torch.cuda.empty_cache()
    return flops / 1e9, sec, achieved / 1e12, mfu, batch


print(f"{'backbone':16s} {'bs':>3s} {'fwd+bwd GFLOP/img':>18s} {'s/iter':>8s} {'achieved TFLOPS':>16s} {'MFU %':>7s}")
for name, factory, cin, batch in build_specs():
    b = batch
    try:
        gf, sec, tf, mfu, b = measure(factory, cin, batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            b = max(8, batch // 2)
            gf, sec, tf, mfu, b = measure(factory, cin, b)
        except Exception as e:
            print(f"{name:16s} OOM ({e})"); torch.cuda.empty_cache(); continue
    per_img = gf / b
    print(f"{name:16s} {b:>3d} {per_img:>18.1f} {sec:>8.3f} {tf:>16.1f} {mfu:>7.1f}")
    torch.cuda.empty_cache()
