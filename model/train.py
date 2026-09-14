"""Train Mod-SwinIR on RealSR V3 (Align L1 + EMA)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import RealSRPairDataset, build_realsr_index
from .losses import OffsetAlignedLoss, psnr
from .metrics import modcrop, official_pair_metrics
from .model import build_model
from .optim import CompositeOptimizer, build_optimizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=r"G:\RealSR\data\RealSR(V3)")
    p.add_argument("--out", type=str, default=r"G:\RealSR\experiments\E11")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--cameras", type=str, default="Canon,Nikon")
    p.add_argument("--model-size", type=str, default="base")
    p.add_argument("--lr-patch", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--align-loss", action="store_true", default=True)
    p.add_argument("--no-align-loss", action="store_true")
    p.add_argument("--max-shift", type=float, default=3.0)
    p.add_argument("--w-pure", type=float, default=0.25)
    p.add_argument("--w-off", type=float, default=0.01)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=1500)
    p.add_argument("--eval-pairs", type=int, default=6)
    p.add_argument("--eval-tile", type=int, default=128)
    p.add_argument("--eval-border", type=int, default=0, help="shave border px (official often uses scale)")
    p.add_argument("--save-every", type=int, default=3000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(s):
    import random

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def run_eval(model, data_root, scale, cameras, max_pairs=6, tile=128, border=0):
    """Fast subset eval with official Y metrics (limited-range, uint8)."""
    from .eval import load_rgb_u8, tiled_forward, to_tensor

    model.eval()
    pairs = build_realsr_index(data_root, cameras, "Test", (scale,))
    if not pairs:
        return {"psnr": 0.0, "ssim": 0.0, "psnr_y": 0.0, "ssim_y": 0.0, "n": 0}
    st = max(1, len(pairs) // max_pairs)
    pairs = pairs[::st][:max_pairs]
    rows = []
    for lr_path, hr_path, sc in pairs:
        lr_u8 = modcrop(load_rgb_u8(lr_path), 4)
        hr_u8 = modcrop(load_rgb_u8(hr_path), 4)
        h, w = lr_u8.shape[:2]
        hr_u8 = hr_u8[: h * sc, : w * sc]
        sr_u8 = tiled_forward(model, to_tensor(lr_u8), sc, tile=tile)
        hh = min(sr_u8.shape[0], hr_u8.shape[0])
        ww = min(sr_u8.shape[1], hr_u8.shape[1])
        rows.append(official_pair_metrics(sr_u8[:hh, :ww], hr_u8[:hh, :ww], with_rgb_ssim=False))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()
    return {
        "psnr": float(np.mean([r["psnr_rgb"] for r in rows])),
        "ssim": float(np.nanmean([r["ssim_rgb"] for r in rows])),
        "psnr_y": float(np.mean([r["psnr_y"] for r in rows])),
        "ssim_y": float(np.mean([r["ssim_y"] for r in rows])),
        "n": len(rows),
        "border": 0,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_align = not args.no_align_loss
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    ds = RealSRPairDataset(
        args.data_root, "Train", cameras, (args.scale,), args.lr_patch, True, None
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    model = build_model(upscale=args.scale, size=args.model_size).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ModSwinIR-{args.model_size} {n_params:.2f}M  pairs={len(ds)}  align={use_align}")

    if use_align:
        criterion = OffsetAlignedLoss(args.max_shift, args.w_pure, args.w_off).to(device)
        extra = list(criterion.offset_net.parameters())
    else:
        criterion = None
        extra = []

    if args.optimizer == "muon":
        # wrap so Muon sees ndim>=2 weights from model + aligner together
        class _Bag(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                if use_align:
                    self.offset_net = criterion.offset_net

        optim = build_optimizer(_Bag(), name="muon", lr=args.lr, weight_decay=1e-4)
    else:
        optim = torch.optim.AdamW(
            list(model.parameters()) + extra, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4
        )
    print(f"optimizer={args.optimizer}")
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    ema = None
    if args.ema > 0:
        import copy

        ema = copy.deepcopy(model).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    def ema_update():
        if ema is None:
            return
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(args.ema).add_(msd[k].detach(), alpha=1 - args.ema)
                else:
                    v.copy_(msd[k])

    step = 0
    best = -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "optim" in ck:
            optim.load_state_dict(ck["optim"])
        step = int(ck.get("step", 0))
        best = float(ck.get("best_psnr", -1))
        if ema is not None:
            ema.load_state_dict(model.state_dict())
        print(f"resume @{step} best={best:.2f}")

    t0 = time.time()
    model.train()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        accum = max(1, int(getattr(args, "grad_accum", 1)))
        is_accum_step = ((step + 1) % accum == 0) or (step == args.steps - 1)
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
            sr, _ = model(lr)
        sr = sr.float()
        if use_align:
            total, parts = criterion(sr, hr)
        else:
            total = F.l1_loss(sr, hr)
            parts = {"l1": total, "plain": total}

        if torch.isfinite(total):
            scaler.scale(total / accum).backward()
            if is_accum_step:
                if isinstance(optim, CompositeOptimizer):
                    for sub in optim.optimizers:
                        scaler.unscale_(sub)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    for sub in optim.optimizers:
                        scaler.step(sub)
                    scaler.update()
                else:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                optim.zero_grad(set_to_none=True)
                ema_update()
        else:
            optim.zero_grad(set_to_none=True)

        if step % 25 == 0 or step == args.steps - 1:
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
            print(
                f"step {step:05d}/{args.steps} | L {float(total.detach()):.4f} "
                f"(l1 {float(parts.get('l1',0)):.4f} aln {float(parts.get('l1_aln',0)):.4f}) "
                f"| lr {cur_lr:.2e} | {mem:.2f}GB | {time.time()-t0:.0f}s",
                flush=True,
            )
            with (out_dir / "train_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "loss": float(total), "lr": cur_lr}) + "\n")

        do_eval = args.eval_every > 0 and ((step + 1) % args.eval_every == 0 or step == args.steps - 1)
        if do_eval:
            net = ema if ema is not None else model
            ev = run_eval(
                net,
                args.data_root,
                args.scale,
                cameras,
                args.eval_pairs,
                args.eval_tile,
                border=args.eval_border,
            )
            print(f"  EVAL {step+1}: RGB {ev['psnr']:.2f}/{ev['ssim']:.4f} Y {ev['psnr_y']:.2f}/{ev['ssim_y']:.4f}")
            with (out_dir / "eval_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, **ev}) + "\n")
            if ev["psnr"] > best:
                best = ev["psnr"]
                torch.save(
                    {
                        "step": step + 1,
                        "model": (ema if ema is not None else model).state_dict(),
                        "optim": optim.state_dict(),
                        "args": vars(args),
                        "best_psnr": best,
                    },
                    out_dir / "ckpt_best.pt",
                )

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save(
                {
                    "step": step + 1,
                    "model": (ema if ema is not None else model).state_dict(),
                    "optim": optim.state_dict(),
                    "args": vars(args),
                    "best_psnr": best,
                },
                out_dir / "ckpt_last.pt",
            )
        step += 1

    print(f"done best={best:.2f} → {out_dir}")


if __name__ == "__main__":
    main()
