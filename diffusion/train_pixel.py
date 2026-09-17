"""Train pixel-space DiT for RealSR x2 (condition = bicubic-up LR).

This is the "pixel" half of the 2x2 diffusion matrix:

* ``flow`` : rectified flow matching. Target is either HR or the residual
             ``(HR - bicubic_up) * 0.5`` (recommended: much smaller dynamic
             range, so the same number of ODE steps buys more accuracy).
* ``reg``  : deterministic residual regressor, ``HR = bicubic_up + f(bicubic_up)``
             with a zero-initialised head, L1 on pixels. This is the arm that is
             directly comparable to the SwinIR regression line (E11, Y 33.47).

HR crop = lr_patch * scale. Keep it at 128 (lr_patch 64) so it matches the
regression line's crop and the eval tile; 256 is feasible on 24G+ with
--grad-ckpt but is 4x the tokens.

Example:
  python -m diffusion.train_pixel --objective flow --lr-patch 64 --batch 8 \
    --steps 20000 --amp --grad-ckpt --out experiments/diffusion/pixel_flow
  python -m diffusion.train_pixel --objective reg --lr-patch 64 --batch 8 \
    --steps 20000 --amp --grad-ckpt --out experiments/diffusion/pixel_reg
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

from .data import RealSRCropDataset, make_split
from .data import default_root
from .dit import build_dit
from .flow import flow_loss, sample_flow
from .metrics import official_pair_metrics
from .vae import psnr01, ssim01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="", help="auto-detected if empty")
    p.add_argument("--out", default=r"G:\RealSR\experiments\diffusion\pixel_dit")
    p.add_argument("--objective", default="flow", choices=["flow", "reg"])
    p.add_argument("--size", default="S", choices=["XS", "S", "M", "B"])
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--heads", type=int, default=0)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--lr-patch", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--t-sampler", default="logit_normal")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--compile", action="store_true", help="torch.compile the training step")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--residual", type=int, default=1,
                   help="flow: model the residual (HR - bicubic_up) instead of HR")
    p.add_argument("--reg-loss", default="l1", choices=["l1", "l2", "smoothl1"])
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


@torch.no_grad()
def run_val(net, val_ds, device, objective, steps, max_n, seed, residual):
    """True Y-PSNR on held-out pairs."""
    net.eval()
    ys, ss = [], []
    n = len(val_ds) if max_n <= 0 else min(len(val_ds), max_n)
    t0 = torch.zeros(1, device=device)
    for i in range(n):
        b = val_ds[i]
        lr = b["lr"].unsqueeze(0).to(device)
        hr = b["hr"].unsqueeze(0).to(device)
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        if objective == "reg":
            rec = (lr_up + net(lr_up, t0)).clamp(0, 1)
        else:
            z = sample_flow(net, lr_up.shape, cond=lr_up, steps=steps, solver="heun", device=device, seed=seed)
            rec = (z * 2.0 + lr_up).clamp(0, 1) if residual else z.clamp(0, 1)
        sr_u8 = (rec[0] * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        hr_u8 = (hr[0] * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        m = official_pair_metrics(sr_u8, hr_u8)
        ys.append(m["psnr_y"])
        ss.append(m["ssim_y"])
    net.train()
    return float(np.mean(ys)), float(np.mean(ss))


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

    ds, val_ds = make_split(
        RealSRCropDataset(args.data_root, "Train", ("Canon", "Nikon"), args.scale, args.lr_patch, True),
        args.val_pairs, seed=args.seed,
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
    is_flow = args.objective == "flow"
    in_ch = 6 if is_flow else 3
    kw = {"input_size": hr_px, "patch_size": args.patch, "in_channels": in_ch, "use_checkpoint": args.grad_ckpt}
    if args.hidden:
        kw["hidden_size"] = args.hidden
    if args.depth:
        kw["depth"] = args.depth
    if args.heads:
        kw["num_heads"] = args.heads
    model = build_dit(args.size, **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"PixelDiT-{args.size} {n_params:.2f}M  objective={args.objective}  HR={hr_px} "
        f"in_ch={in_ch} residual={bool(args.residual)} train={len(ds)} val={0 if val_ds is None else len(val_ds)}",
        flush=True,
    )

    model_raw = model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = None
    if args.ema > 0:
        ema = copy.deepcopy(model).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    if args.compile:
        try:
            model = torch.compile(model_raw, mode=args.compile_mode)
            print(f"torch.compile ON (mode={args.compile_mode})", flush=True)
        except Exception as exc:  # never let a compile failure kill a long run
            model = model_raw
            print(f"torch.compile FAILED ({exc}); falling back to eager", flush=True)

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
        model_raw.load_state_dict(ck["model"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        step = int(ck.get("step", 0))
        best = float(ck.get("best_psnr_y", ck.get("best_psnr01", -1.0)))
        print(f"resumed from {args.resume} step={step} best={best:.3f}", flush=True)

    amp_dtype = torch.bfloat16 if args.amp else torch.float32
    t_zeros = torch.zeros(1, device=device)

    def forward_loss(lr_up, hr):
        if is_flow:
            x0 = (hr - lr_up).clamp(-1, 1) * 0.5 if args.residual else hr
            return flow_loss(model, x0, lr_up, t_mode=args.t_sampler)[0]
        res = model(lr_up, t_zeros.expand(lr_up.shape[0]))
        pred = (lr_up + res).clamp(0, 1)
        if args.reg_loss == "l2":
            return F.mse_loss(pred, hr)
        if args.reg_loss == "smoothl1":
            return F.smooth_l1_loss(pred, hr)
        return F.l1_loss(pred, hr)

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

        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
            loss = forward_loss(lr_up, hr)
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
                f"step {step:06d}/{args.steps} | L {float(loss.detach()):.5f} | "
                f"lr {cur_lr:.2e} | {mem:.2f}GB | {time.time()-t0:.0f}s",
                flush=True,
            )
            with (out / "train_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.detach()), "lr": cur_lr}) + "\n")

        if args.eval_every > 0 and val_ds is not None and ((step + 1) % args.eval_every == 0):
            net = ema if ema is not None else model_raw
            psnr, ssim = run_val(
                net, val_ds, device, args.objective, max(1, args.val_eval_steps),
                args.val_pairs, args.seed, bool(args.residual),
            )
            print(f"  VAL {step+1}: Y {psnr:.3f}/{ssim:.4f} n={min(len(val_ds), args.val_pairs)}", flush=True)
            with (out / "val_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, "psnr_y": psnr, "ssim_y": ssim}) + "\n")
            if psnr > best:
                best = psnr
                no_gain = 0
                torch.save(save_payload(args, model, ema, step + 1, best, hr_px, in_ch), out / "ckpt_best.pt")
            else:
                no_gain += 1
            if no_gain >= args.patience and step + 1 >= args.min_steps:
                print(f"  EARLY STOP at {step+1} (no val gain for {args.patience} evals)", flush=True)
                torch.save(save_payload(args, model, ema, step + 1, best, hr_px, in_ch), out / "ckpt_last.pt")
                break

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save(save_payload(args, model, ema, step + 1, best, hr_px, in_ch), out / "ckpt_last.pt")
        step += 1

    print(f"done best_val_Y={best:.3f} → {out}", flush=True)


def _unwrap(m):
    """torch.compile wraps the module; keep checkpoint keys clean."""
    return getattr(m, "_orig_mod", m)


def save_payload(args, model, ema, step, best, hr_px, in_ch):
    net = ema if ema is not None else _unwrap(model)
    return {
        "step": step,
        "model": net.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "args": vars(args),
        "objective": args.objective,
        "mode": "pixel",
        "hr_px": hr_px,
        "in_channels": in_ch,
        "residual": bool(args.residual),
        "best_psnr_y": best,
        "best_psnr01": best,
    }


if __name__ == "__main__":
    main()
