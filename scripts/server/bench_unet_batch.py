"""Size the U-Net batch to fill ~40 GB of the 4090, then time it.

The user's target is 40 GB: measured earlier that this card goes super-linear
past ~40 GB, so 40 is the sweet spot, not 48.

Stage 1  build the model, assert the zero-init identity property, then sweep
         batch sizes with a single fwd+bwd each and report peak memory.
         (One step is enough to size memory, and it avoids recompiling.)
Stage 2  for the chosen batch, run a timed loop to get s/step and samples/s.
"""
import gc
import sys, time, argparse
sys.path.insert(0, "/home/ds/realsr")

import torch
from diffusion.unet import build_unet

HR = 128
IN_CH = 3


def build(base, mult, num_res, attn_levels, in_stride=1, out_scale=1):
    return build_unet("S", input_size=HR, in_channels=IN_CH, base=base,
                      mult=mult, num_res=num_res, attn_levels=attn_levels,
                      in_stride=in_stride, out_scale=out_scale).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--mult", default="1,2,4,4")
    ap.add_argument("--num-res", type=int, default=2)
    ap.add_argument("--attn-levels", default="2,3")
    ap.add_argument("--batches", default="64,128,192,256,320,384")
    ap.add_argument("--target-gb", type=float, default=40.0)
    ap.add_argument("--timed", type=int, default=0, help="if >0, run a timed loop at this batch")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--in-stride", type=int, default=1)
    ap.add_argument("--out-scale", type=int, default=1)
    args = ap.parse_args()

    mult = tuple(int(m) for m in args.mult.split(",") if m.strip())
    attn = tuple(int(m) for m in args.attn_levels.split(",") if m.strip())

    m = build(args.base, mult, args.num_res, attn, args.in_stride, args.out_scale)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"UNet base={args.base} mult={mult} num_res={args.num_res} attn={attn} "
          f"in_stride={args.in_stride} out_scale={args.out_scale} "
          f"HR={HR} -> {n_params:.2f}M params, align={m.align}", flush=True)

    # zero-init identity: a fresh net must output exactly 0 (so reg starts at bicubic)
    with torch.no_grad():
        x = torch.randn(2, IN_CH, HR, HR, device="cuda")
        t = torch.zeros(2, device="cuda")
        out = m(x, t)
    print(f"identity check: max|out| = {out.abs().max().item():.3e} "
          f"({'OK' if out.abs().max().item() < 1e-6 else 'FAIL'})", flush=True)
    del x, out, m
    torch.cuda.empty_cache()

    print(f"\n{'BATCH':>6} {'PEAK_GB':>9} {'TOTAL_GB':>9}", flush=True)
    chosen = None
    for b in [int(v) for v in args.batches.split(",") if v.strip()]:
        m = opt = x = t = loss = None
        gc.collect()
        torch.cuda.empty_cache()
        m = build(args.base, mult, args.num_res, attn, args.in_stride, args.out_scale)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x = torch.randn(b, IN_CH, HR, HR, device="cuda")
                t = torch.zeros(b, device="cuda")
                loss = m(x, t).float().pow(2).mean()
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            total = torch.cuda.memory_reserved() / 1024**3
            print(f"{b:>6} {peak:>9.2f} {total:>9.2f}", flush=True)
            if peak <= args.target_gb:
                chosen = b
        except torch.cuda.OutOfMemoryError:
            print(f"{b:>6} {'OOM':>9}", flush=True)
        del m, opt, x, t, loss
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nlargest batch under {args.target_gb} GB: {chosen}", flush=True)

    if args.timed:
        b = args.timed
        m = build(args.base, mult, args.num_res, attn, args.in_stride, args.out_scale)
        if args.compile:
            m = torch.compile(m)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randn(b, IN_CH, HR, HR, device="cuda")
        t = torch.zeros(b, device="cuda")
        for i in range(4):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, t).float().pow(2).mean()
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        n = 20
        t0 = time.time()
        for _ in range(n):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, t).float().pow(2).mean()
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n
        print(f"\nTIMED batch={b} compile={bool(args.compile)}: {dt:.3f} s/step "
              f"-> {b/dt:.1f} samples/s, peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GB",
              flush=True)


if __name__ == "__main__":
    main()
