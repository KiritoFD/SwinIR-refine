"""Train latent DiT for RealSR x2 (condition on bicubic-up LR latent).

Two objectives, same backbone, same data, same crop size — this is the
"latent" half of the 2x2 diffusion matrix:

* ``flow``  : rectified flow matching (logit-normal t, Heun ODE at eval).
              Stochastic, needs N integration steps.
* ``reg``   : deterministic residual regressor in latent space,
              z0_hat = z_cond + f(z_cond) with a zero-initialised final layer
              (starts as "identity = bicubic"). This is the PSNR-optimal arm:
              it directly predicts the conditional mean instead of sampling it.

Example (server 48G):
  python -m diffusion.train_latent --objective flow --vae flux1-vae \
    --lr-patch 128 --batch 32 --steps 20000 --amp \
    --out experiments/diffusion/latent_flow

  python -m diffusion.train_latent --objective reg --vae flux1-vae \
    --lr-patch 128 --batch 32 --steps 20000 --amp \
    --out experiments/diffusion/latent_reg
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import RealSRCropDataset
from .dit import build_dit
from .flow import flow_loss, sample_flow
from .vae import decode, decode_grad, encode, load_vae, psnr01, ssim01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"G:\RealSR\data\RealSR(V3)")
    p.add_argument("--out", default=r"G:\RealSR\experiments\diffusion\latent_dit")
    p.add_argument("--objective", default="flow", choices=["flow", "reg"])
    p.add_argument("--vae", default="flux1-vae")
    p.add_argument("--vae-dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--size", default="S", choices=["S", "M", "B"])
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--hidden", type=int, default=0, help="0 = model default")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--heads", type=int, default=0)
    p.add_argument("--scale", type=int, default=2)
    # HR crop = lr_patch * scale. Keep it equal to the eval HR tile so train and
    # inference see the same token grid. 128 -> HR256 -> latent 32x32 (f8).
    p.add_argument("--lr-patch", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--t-sampler", default="logit_normal", choices=["logit_normal", "uniform", "cosmap"])
    p.add_argument("--ema", type=float, default=0.999, help="0 disables; 0.999 fits 20k steps")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--amp", action="store_true")
    # deterministic target: use the posterior mode, never sample the HR latent.
    p.add_argument("--sample-hr", type=int, default=0)
    p.add_argument("--reg-loss", default="l1", choices=["l1", "l2", "smoothl1"])
    p.add_argument("--pixel-loss-weight", type=float, default=0.0,
                   help="reg only: add L1 on decoded pixels (grad flows through frozen VAE)")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-pairs", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=20, help="flow only: ODE steps")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    vae_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.vae_dtype]
    vae, vinfo = load_vae(args.vae, device=device, dtype=vae_dtype)
    ds = RealSRCropDataset(
        args.data_root, "Train", ("Canon", "Nikon"), args.scale, args.lr_patch, True, None
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    hr_px = args.lr_patch * args.scale
    assert hr_px % vinfo.downscale == 0, f"HR {hr_px} not divisible by VAE ds {vinfo.downscale}"
    lat = hr_px // vinfo.downscale
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
        f"LatentDiT-{args.size} {n_params:.2f}M  objective={args.objective}  "
        f"HR={hr_px} latent {lat}x{lat}x{vinfo.latent_channels} in_ch={in_ch}  "
        f"VAE={vinfo.name}[{vinfo.norm_tag}]  pairs={len(ds)}",
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
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ema is not None and "ema" in ck:
            ema.load_state_dict(ck["ema"])
        step = int(ck.get("step", 0))
        best = float(ck.get("best_psnr01", -1.0))
        print(f"resumed from {args.resume} step={step} best={best:.3f}", flush=True)

    amp_dtype = torch.bfloat16 if args.amp else torch.float32
    t_zeros = torch.zeros(1, device=device)

    def forward_loss(z0, zc, hr):
        """z0/zc are normalized latents; hr is the rgb target in [0,1] (reg+pixel-loss only)."""
        if is_flow:
            return flow_loss(model, z0, zc, t_mode=args.t_sampler)[0]
        res = model(zc, t_zeros.expand(zc.shape[0]))
        z0_hat = zc + res
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
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr

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

        if step % 50 == 0 or step == args.steps - 1:
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
            print(
                f"step {step:06d}/{args.steps} | L {float(loss.detach()):.4f} | "
                f"lr {cur_lr:.2e} | {mem:.2f}GB | {time.time()-t0:.0f}s",
                flush=True,
            )
            with (out / "train_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.detach()), "lr": cur_lr}) + "\n")

        if args.eval_every > 0 and ((step + 1) % args.eval_every == 0 or step == args.steps - 1):
            net = ema if ema is not None else model
            net.eval()
            b = next(iter(loader))
            lrb = b["lr"][: args.eval_pairs].to(device)
            hrb = b["hr"][: args.eval_pairs].to(device)
            lr_up = F.interpolate(lrb, size=hrb.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
            with torch.no_grad():
                zc = encode(vae, lr_up, vinfo, sample=False)
                if is_flow:
                    z = sample_flow(
                        net,
                        (zc.shape[0], vinfo.latent_channels, lat, lat),
                        cond=zc,
                        steps=args.eval_steps,
                        solver="heun",
                        device=device,
                        seed=args.seed,
                    )
                else:
                    z = zc + net(zc, t_zeros.expand(zc.shape[0]))
                rec = decode(vae, z, vinfo)
                ps = [psnr01(rec[i : i + 1], hrb[i : i + 1]) for i in range(rec.shape[0])]
                ss = [ssim01(rec[i : i + 1], hrb[i : i + 1]) for i in range(rec.shape[0])]
            ev = {"psnr01": float(sum(ps) / len(ps)), "ssim01": float(sum(ss) / len(ss)), "n": len(ps)}
            print(
                f"  EVAL {step+1}: RGB01 {ev['psnr01']:.2f}/{ev['ssim01']:.4f} n={ev['n']}",
                flush=True,
            )
            with (out / "eval_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, **ev}) + "\n")
            if ev["psnr01"] > best:
                best = ev["psnr01"]
                torch.save(save_payload(args, model, ema, step + 1, best, lat, in_ch, vinfo), out / "ckpt_best.pt")
            model.train()

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save(save_payload(args, model, ema, step + 1, best, lat, in_ch, vinfo), out / "ckpt_last.pt")
        step += 1

    print(f"done best_psnr01={best:.2f} → {out}", flush=True)


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
        "best_psnr01": best,
    }


if __name__ == "__main__":
    main()
