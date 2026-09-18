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
from torch.utils.data import DataLoader, RandomSampler

from .data import RealSRCropDataset, make_split
from .data import default_root
from .dit import build_dit
from .unet import build_unet
from .flow import flow_loss, sample_flow
from .metrics import official_pair_metrics
from .vae import psnr01, ssim01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="", help="auto-detected if empty")
    p.add_argument("--out", default=r"G:\RealSR\experiments\diffusion\pixel_dit")
    p.add_argument("--objective", default="flow", choices=["flow", "reg"])
    p.add_argument("--backbone", default="dit", choices=["dit", "unet"],
                   help="dit = token transformer; unet = conv encoder/decoder with skips")
    p.add_argument("--size", default="S", choices=["XS", "S", "M", "B"])
    p.add_argument("--patch", type=int, default=2)
    # U-Net only (ignored by the DiT path)
    p.add_argument("--base", type=int, default=0, help="unet: base channel count (0 = from --size)")
    p.add_argument("--mult", default="1,2,4,4", help="unet: channel multiplier per level")
    p.add_argument("--num-res", type=int, default=2, help="unet: residual blocks per level")
    p.add_argument("--attn-levels", default="2,3", help="unet: levels that get self-attention")
    p.add_argument("--native-lr", type=int, default=-1,
                   help="unet: run the encoder/decoder at LR scale and pixel-shuffle back up. "
                        "-1 auto (on for reg, off for flow), 0 off, 1 on")
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
    p.add_argument("--cache-data", type=int, default=1,
                   help="preload and decode the training images once; without it the PNG "
                        "decode caps the loader at a few hundred samples/s")
    p.add_argument("--decoded-manifest", default="data/decoded/manifest.json",
                   help="manifest of pre-decoded uint8 blobs (scripts/server/precache_hr.py); "
                        "images found there are read via mmap instead of decoded. '' disables.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--init", default="",
                   help="load weights only (step/optimiser reset) — for fine-tuning a pretrained ckpt")
    p.add_argument("--pretrain-root", default="",
                   help="HR image dir for SwinIR/BSRGAN-style synthetic pretraining")
    p.add_argument("--pretrain-limit", type=int, default=0, help="cap pretraining images (0 = all)")
    p.add_argument("--pretrain-val-root", default="",
                   help="held-out images for the PRETRAIN val (must NOT overlap --pretrain-root). "
                        "Falls back to the RealSR split, which measures domain transfer rather "
                        "than pretraining progress.")
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


def _worker_init(_worker_id: int):
    """Pin each DataLoader worker to a single compute thread.

    Left alone, every worker spawns its own OpenMP pool -- 24 threads each on
    this box, so 12 workers means ~288 spinning threads competing for 24 cores.
    BSRGAN's degradation is a handful of tiny 128x128 convolutions, and that
    thrash made it 10x slower than single-threaded: _blur 33.0 ms -> 3.3 ms,
    the full degrade 153 ms -> 10.6 ms per item.  That difference is the whole
    reason the loader could not feed a 768-sample batch.
    """
    torch.set_num_threads(1)


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

    dm = args.decoded_manifest or None
    _real = RealSRCropDataset(args.data_root, "Train", ("Canon", "Nikon"), args.scale, args.lr_patch, True,
                             cache=bool(args.cache_data), decoded_manifest=dm)
    if args.pretrain_root:
        # train on BSRGAN-degraded DIV2K/Flickr2K, but KEEP the RealSR val split
        # so we can watch zero-shot transfer while pretraining.
        from .bsrgan import BSRGANDataset

        ds = BSRGANDataset(args.pretrain_root, args.lr_patch, args.scale, True,
                           args.pretrain_limit, args.seed, decoded_manifest=dm)
        # Validate on HELD-OUT images from the pretrain domain, degraded the same
        # way.  Watching the RealSR val split during pretraining measures domain
        # transfer, not whether pretraining is working -- and since it drove
        # ckpt_best, the fine-tune was being initialised from whichever step
        # happened to transfer best, i.e. noise.
        if args.pretrain_val_root:
            val_ds = BSRGANDataset(args.pretrain_val_root, args.lr_patch, args.scale, False,
                                   0, args.seed, decoded_manifest=dm, deterministic=True)
        else:
            _, val_ds = make_split(_real, args.val_pairs, seed=args.seed)
        print(f"pretrain: {len(ds)} HR images from {args.pretrain_root}; "
              f"val {0 if val_ds is None else len(val_ds)} from "
              f"{args.pretrain_val_root or 'RealSR split (NOT recommended)'}", flush=True)
    else:
        ds, val_ds = make_split(_real, args.val_pairs, seed=args.seed)
    # RealSR Train has only 390 usable pairs.  With batch >= len(ds) an epoch
    # yields zero (drop_last) or one full batch, and the loop would rebuild the
    # worker pool on almost every step.  Draw several batches per epoch with
    # replacement instead, and keep the workers alive across epochs.
    sampler = None
    if len(ds) <= args.batch:
        sampler = RandomSampler(ds, replacement=True, num_samples=args.batch * 8)
        print(f"  sampler: {len(ds)} pairs < batch {args.batch} -> with replacement, "
              f"{args.batch * 8} samples/epoch ({8} steps)", flush=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init if args.num_workers > 0 else None,
    )
    hr_px = args.lr_patch * args.scale
    is_flow = args.objective == "flow"
    in_ch = 6 if is_flow else 3
    kw = {"input_size": hr_px, "patch_size": args.patch, "in_channels": in_ch, "use_checkpoint": args.grad_ckpt}
    if args.backbone == "unet":
        kw.pop("patch_size")
        kw["mult"] = tuple(int(m) for m in args.mult.split(",") if m.strip())
        kw["num_res"] = args.num_res
        kw["attn_levels"] = tuple(int(m) for m in args.attn_levels.split(",") if m.strip())
        if args.base:
            kw["base"] = args.base
        ns = args.native_lr
        if ns < 0:
            ns = 1 if args.objective == "reg" else 0
        kw["in_stride"] = 2 if ns else 1
        kw["out_scale"] = 2 if ns else 1
        model = build_unet(args.size, **kw).to(device)
        print(f"  unet native_lr={bool(ns)} in_stride={kw['in_stride']} "
              f"out_scale={kw['out_scale']} align={model.align}", flush=True)
    else:
        if args.hidden:
            kw["hidden_size"] = args.hidden
        if args.depth:
            kw["depth"] = args.depth
        if args.heads:
            kw["num_heads"] = args.heads
        model = build_dit(args.size, **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"{'PixelUNet' if args.backbone == 'unet' else 'PixelDiT'}-{args.size} {n_params:.2f}M  "
        f"backbone={args.backbone} objective={args.objective}  HR={hr_px} "
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

    if args.init and Path(args.init).is_file():
        # weights only: fine-tune a pretrained ckpt from step 0 with a fresh schedule
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model_raw.load_state_dict(ck["model"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        step, best, no_gain = 0, -1.0, 0
        print(f"init weights from {args.init} (training state reset)", flush=True)

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
