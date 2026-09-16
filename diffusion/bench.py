"""Benchmark every arm of the diffusion matrix before committing to a schedule.

Prints ms/step, peak VRAM and forward-only ms so the 36h plan can be built from
measured numbers instead of guesses.

  python -m diffusion.bench --iters 8
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from .dit import build_dit
from .flow import flow_loss, sample_flow
from .vae import decode, encode, load_vae

VAE_BF16 = True

CONFIGS = [
    # tag,                      mode,     vae,          objective, size, lr_patch, batch
    ("latent_flow_flux256", "latent", "flux1-vae", "flow", "S", 256, 32),
    ("latent_reg_flux256", "latent", "flux1-vae", "reg", "S", 256, 32),
    ("latent_flow_flux128", "latent", "flux1-vae", "flow", "S", 128, 64),
    ("latent_flow_sd256", "latent", "sd-vae-ft-ema", "flow", "S", 256, 32),
    ("latent_flow_sdxl256", "latent", "sdxl-vae", "flow", "S", 256, 32),
    ("latent_flow_flux256_XS", "latent", "flux1-vae", "flow", "XS", 256, 32),
    ("pixel_flow128", "pixel", "", "flow", "S", 64, 16),
    ("pixel_reg128", "pixel", "", "reg", "S", 64, 16),
    ("pixel_flow128_XS", "pixel", "", "flow", "XS", 64, 16),
    ("pixel_reg128_XS", "pixel", "", "reg", "XS", 64, 16),
    ("pixel_flow256", "pixel", "", "flow", "S", 128, 4),
]


def one(tag, mode, vae_name, objective, size, lr_patch, batch, iters, warm, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vae = vinfo = None
    if mode == "latent":
        vae, vinfo = load_vae(vae_name, device=device, dtype=torch.bfloat16 if VAE_BF16 else torch.float32)
        hr = lr_patch * 2
        lat = hr // vinfo.downscale
        c = vinfo.latent_channels
        input_size = lat
        in_ch = 2 * c if objective == "flow" else c
        x0 = torch.randn(batch, c, lat, lat, device=device)
        hr_t = torch.rand(batch, 3, hr, hr, device=device)
        cond = torch.rand(batch, 3, hr, hr, device=device)
    else:
        hr = lr_patch * 2
        input_size = hr
        in_ch = 6 if objective == "flow" else 3
        x0 = torch.randn(batch, 3, hr, hr, device=device)
        cond = torch.randn(batch, 3, hr, hr, device=device)
        hr_t = None

    model = build_dit(size, input_size=input_size, patch_size=2, in_channels=in_ch).to(device)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    t_zero = torch.zeros(batch, device=device)
    hold = {}

    def step():
        if mode == "latent":
            with torch.no_grad():
                z0 = encode(vae, hr_t, vinfo, sample=False)
                zc = encode(vae, cond, vinfo, sample=False)
                hold["zc"] = zc
        if objective == "flow":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = flow_loss(model, z0, zc)[0] if mode == "latent" else flow_loss(model, x0, cond)[0]
            loss.backward()
        else:
            if mode == "latent":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    zh = zc + model(zc, t_zero)
                    loss = F.l1_loss(zh, z0)
                    rec = decode(vae, zh, vinfo)
                loss.backward()
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.l1_loss((cond + model(cond, t_zero)).clamp(0, 1), x0)
                loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    try:
        for i in range(warm):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(iters):
            step()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / iters * 1000
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(iters):
                cin = hold.get("zc", cond) if mode == "latent" else cond
                if objective == "reg":
                    model(cin, t_zero)
                else:
                    model(torch.cat([x0, cin], 1), t_zero)
        torch.cuda.synchronize()
        fwd_ms = (time.time() - t0) / iters * 1000
        mem = torch.cuda.max_memory_allocated() / 1024**3
        tokens = (input_size // 2) ** 2 * batch
        out = dict(tag=tag, params_M=round(npar, 2), batch=batch, tokens=tokens,
                   ms_step=round(ms, 1), ms_fwd=round(fwd_ms, 2), peak_GB=round(mem, 2), ok=True)
    except RuntimeError as e:
        out = dict(tag=tag, batch=batch, ok=False, err=str(e)[:80])
    del model, opt
    if vae is not None:
        del vae
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--only", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()
    print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    rows = []
    for cfg in CONFIGS:
        if args.only and args.only not in cfg[0]:
            continue
        r = one(*cfg, iters=args.iters, warm=args.warm)
        rows.append(r)
        if r["ok"]:
            print(f"{r['tag']:24s} {r['params_M']:6.2f}M b={r['batch']:3d} "
                  f"tok={r['tokens']:7d} {r['ms_step']:8.1f}ms/step  fwd {r['ms_fwd']:7.2f}ms  "
                  f"peak {r['peak_GB']:5.2f}GB", flush=True)
        else:
            print(f"{r['tag']:24s} FAILED {r['err']}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
