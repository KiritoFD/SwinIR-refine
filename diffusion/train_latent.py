"""Train latent DiT for RealSR x2 (condition on bicubic-up LR latent).

Two objectives, same backbone, same data, same crop size — this is the
"latent" half of the 2x2 diffusion matrix:

* ``flow``  : rectified flow matching (logit-normal t, Heun ODE at eval).
* ``reg``   : deterministic residual regressor in latent space,
              z0_hat = z_cond + f(z_cond), zero-initialised head.

Fast path: with ``--latent-cache`` (see ``diffusion.precompute_latents``) the VAE
is not touched during training — the DiT step drops from ~1.18 s to ~0.17 s.

Example (4090 48G):
  python -m diffusion.train_latent --objective flow --vae flux1-vae \
    --latent-cache data/latents/flux1-vae --lr-patch 256 --batch 32 \
    --steps 40000 --amp --out experiments/diffusion/latent_flow_flux
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import LatentCropDataset, RealSRCropDataset, make_split
from .data import default_root
from .dit import build_dit
from .flow import flow_loss, sample_flow
from .metrics import official_pair_metrics
from .vae import decode, decode_grad, encode, load_vae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="", help="auto-detected if empty")
    p.add_argument("--out", default=r"G:\RealSR\experiments\diffusion\latent_dit")
    p.add_argument("--objective", default="flow", choices=["flow", "reg"])
    p.add_argument("--vae", default="flux1-vae")
    p.add_argument("--vae-dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--latent-cache", default="")
    p.add_argument("--size", default="S", choices=["XS", "S", "M", "B"])
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--heads", type=int, default=0)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--lr-patch", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--t-sampler", default="logit_normal", choices=["logit_normal", "uniform", "cosmap"])
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--sample-hr", type=int, default=0)
    p.add_argument("--reg-loss", default="l1", choices=["l1", "l2", "smoothl1"])
    p.add_argument("--pixel-loss-weight", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--val-pairs", type=int, default=16)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-steps", type=int, default=4000)
    p.add_argument("--val-eval-steps", type=int, default=8)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not getattr(args, "data_root", ""):
        args.data_root = default_root()
    return args


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def run_val(net, val_ds, vae, vinfo, device, objective, steps, max_n, seed):
    """True Y-PSNR on held-out pairs (raw images, so each item costs a VAE pass)."""
    net.eval()
    ys, ss = [], []
    n = len(val_ds) if max_n <= 0 else min(len(val_ds), max_n)
    t0 = torch.zeros(1, device=device)
    for i in range(n):
        b = val_ds[i]
        lr = b["lr"].unsqueeze(0).to(device)
        hr = b["hr"].unsqueeze(0).to(device)
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        zc = encode(vae, lr_up, vinfo, sample=False)
        if objective == "reg":
            z = zc + net(zc, t0)
        else:
            z = sample_flow(net, zc.shape, cond=zc, steps=steps, solver="heun", device=device, seed=seed)
        rec = decode(vae, z, vinfo)
        sr_u8 = (rec[0].clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        hr_u8 = (hr[0] * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        m = official_pair_metrics(sr_u8, hr_u8)
        ys.append(m["psnr_y"])
        ss.append(m["ssim_y"])
    net.train()
    return float(np.mean(ys)), float(np.mean(ss))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    vae_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.vae_dtype]
    vae, vinfo = load_vae(args.vae, device=device, dtype=vae_dtype)
    hr_px = args.lr_patch * args.scale
    assert hr_px % vinfo.downscale == 0, f"HR {hr_px} not divisible by VAE ds {vinfo.downscale}"
    lat = hr_px // vinfo.downscale

    base = RealSRCropDataset(args.data_root, "Train", ("Canon", "Nikon"), args.scale, args.lr_patch, True)
    train_base, val_ds = make_split(base, args.val_pairs, seed=args.seed)
    val_names = set()
    if val_ds is not None:
        val_names = {Path(p[0]).stem for p in val_ds.pairs}

    if args.latent_cache:
        train_ds = LatentCropDataset(args.latent_cache, crop=lat, augment=True)
        if val_names:
            train_ds.files = [f for f in train_ds.files if f.stem not in val_names]
        print(f"latent cache: {len(train_ds)} train pairs, {len(val_names)} held out", flush=True)
    else:
        train_ds = train_base

    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    is_flow = args.objective == "flow"
    in_ch = 2 * vinfo.latent_channels if is_flow else vinfo.latent_channels
    kw = {"input_size": lat, "patch_size": args.patch, "in_channels": in_ch, "use_checkpoint": args.grad_ckpt}
    if args.hidden:
        kw["hidden_size"] = args.hidden
    if args.depth:
        kw["depth"] = args.depth
    if args.heads:
        kw["num_heads"] = args.heads
    model = build_dit(args.size, **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"LatentDiT-{args.size} {n_params:.2f}M  objective={args.objective}  HR={hr_px} "
        f"latent {lat}x{lat}x{vinfo.latent_channels} in_ch={in_ch}  VAE={vinfo.name}[{vinfo.norm_tag}]  "
        f"train={len(train_ds)} val={0 if val_ds is None else len(val_ds)} cache={bool(args.latent_cache)}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = None
    if args.ema > 0:
        ema = copy.deepcopy(model).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    def ema_update():
        if ema is None:
            return
        with torch.no_grad():
            for a, b in zip(ema.parameters(), model.parameters()):
                a.mul_(args.ema).add_(b.detach(), alpha=1 - args.ema)

    step = 0
    best = -1.0
    no_gain = 0
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        step = int(ck.get("step", 0))
        best = float(ck.get("best_psnr_y", -1.0))
        print(f"resumed from {args.resume} step={step} best={best:.3f}", flush=True)

    amp_dtype = torch.bfloat16 if args.amp else torch.float32
    t_zeros = torch.zeros(1, device=device)

    def forward_loss(z0, zc, hr):
        if is_flow:
            return flow_loss(model, z0, zc, t_mode=args.t_sampler)[0]
        z0_hat = zc + model(zc, t_zeros.expand(zc.shape[0]))
        if args.reg_loss == "l2":
            loss = F.mse_loss(z0_hat, z0)
        elif args.reg_loss == "smoothl1":
            loss = F.smooth_l1_loss(z0_hat, z0)
        else:
            loss = F.l1_loss(z0_hat, z0)
        if args.pixel_loss_weight > 0:
            rec = decode_grad(vae, z0_hat, vinfo)
            loss = loss + args.pixel_loss_weight * F.l1_loss(rec, hr)
        return loss

    t0 = time.time()
    it = iter(loader)
    model.train()
    stopped = False
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        if args.latent_cache:
            z0 = batch["z0"].to(device, non_blocking=True)
            zc = batch["zc"].to(device, non_blocking=True)
            hr = None
        else:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            with torch.no_grad():
                lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
                z0 = encode(vae, hr, vinfo, sample=bool(args.sample_hr))
                zc = encode(vae, lr_up, vinfo, sample=False)

        with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
            loss = forward_loss(z0, zc, hr)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        ema_update()

        if step % 100 == 0 or step == args.steps - 1:
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
            print(
                f"step {step:06d}/{args.steps} | L {float(loss.detach()):.4f} | lr {cur_lr:.2e} | "
                f"{mem:.2f}GB | {time.time()-t0:.0f}s",
                flush=True,
            )
            with (out / "train_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.detach()), "lr": cur_lr}) + "\n")

        if args.eval_every > 0 and val_ds is not None and ((step + 1) % args.eval_every == 0):
            net = ema if ema is not None else model
            psnr, ssim = run_val(
                net, val_ds, vae, vinfo, device, args.objective,
                max(1, args.val_eval_steps), args.val_pairs, args.seed,
            )
            print(f"  VAL {step+1}: Y {psnr:.3f}/{ssim:.4f} n={min(len(val_ds), args.val_pairs)}", flush=True)
            with (out / "val_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, "psnr_y": psnr, "ssim_y": ssim}) + "\n")
            if psnr > best:
                best = psnr
                no_gain = 0
                torch.save(save_payload(args, model, ema, step + 1, best, lat, in_ch, vinfo), out / "ckpt_best.pt")
            else:
                no_gain += 1
            if no_gain >= args.patience and step + 1 >= args.min_steps:
                print(f"  EARLY STOP at {step+1} (no val gain for {args.patience} evals)", flush=True)
                torch.save(save_payload(args, model, ema, step + 1, best, lat, in_ch, vinfo), out / "ckpt_last.pt")
                stopped = True

        if (step + 1) % args.save_every == 0:
            torch.save(save_payload(args, model, ema, step + 1, best, lat, in_ch, vinfo), out / "ckpt_last.pt")
        if stopped:
            break
        step += 1

    if not stopped:
        torch.save(save_payload(args, model, ema, step, best, lat, in_ch, vinfo), out / "ckpt_last.pt")
    print(f"done best_val_Y={best:.3f} -> {out}", flush=True)


def save_payload(args, model, ema, step, best, lat, in_ch, vinfo):
    net = ema if ema is not None else model
    return {
        "step": step,
        "model": net.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "args": vars(args),
        "objective": args.objective,
        "mode": "latent",
        "vae": vinfo.name,
        "vae_shift_factor": vinfo.shift_factor,
        "vae_scaling_factor": vinfo.scaling_factor,
        "latent_size": lat,
        "in_channels": in_ch,
        "hr_px": args.lr_patch * args.scale,
        "best_psnr_y": best,
        "best_psnr01": best,
    }


if __name__ == "__main__":
    main()
