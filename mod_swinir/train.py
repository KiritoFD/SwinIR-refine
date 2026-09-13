"""Train SwinIR / Mod-SwinIR on RealSR V3 (periodic eval, NaN-safe)."""

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
from .losses import UWCLLoss, psnr
from .model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="RealSR V3 SR training (SwinIR / ModSwinIR)")
    p.add_argument("--data-root", type=str, default=r"G:\RealSR\data\RealSR(V3)")
    p.add_argument("--out", type=str, default=r"G:\RealSR\experiments\run")
    p.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--cameras", type=str, default="Canon,Nikon")
    p.add_argument(
        "--arch",
        type=str,
        default="mod",
        choices=["mod", "swinir"],
        help="mod=ModSwinIR, swinir=official baseline",
    )
    p.add_argument("--model-size", type=str, default="base", help="mod: tiny|base|large | swinir: light|mid|classical|largeish")
    p.add_argument("--lr-patch", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-pairs", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-pairs", type=int, default=4)
    p.add_argument("--eval-tile", type=int, default=128)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w-ctx", type=float, default=0.1)
    p.add_argument("--w-plain", type=float, default=0.5)
    p.add_argument("--w-grad", type=float, default=0.0)
    p.add_argument("--w-amp", type=float, default=0.05, help="patch-wise amplitude loss weight")
    p.add_argument("--w-hf-conf", type=float, default=0.25, help="confidence-weighted HF loss weight")
    p.add_argument("--align-loss", action="store_true", help="offset-aligned L1 (mimicked alignment)")
    p.add_argument("--max-shift", type=float, default=3.0)
    p.add_argument("--w-pure", type=float, default=0.25)
    p.add_argument("--w-off", type=float, default=0.01)
    p.add_argument("--use-uwcl", action="store_true")
    p.add_argument("--l1-only", action="store_true", help="plain L1 for any arch (ablation)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay (0=off, e.g. 0.999)")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--nan-lr-scale", type=float, default=0.5, help="halve LR after consecutive NaNs")
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


def build_any(args):
    n_params = 0
    if args.arch == "mod":
        model = build_model(upscale=args.scale, size=args.model_size)
        tag = f"ModSwinIR-{args.model_size}"
    else:
        from .swinir_baseline import SwinIRTrainWrapper, build_swinir

        model = SwinIRTrainWrapper(build_swinir(args.model_size, upscale=args.scale))
        tag = f"SwinIR-{args.model_size}"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return model, tag, n_params


@torch.no_grad()
def run_eval(model, data_root, scale, cameras, max_pairs=4, tile=128):
    from .eval import load_png, psnr_torch, rgb_to_y_torch, ssim_torch, tiled_forward, to_tensor

    model.eval()
    pairs = build_realsr_index(data_root, cameras, "Test", (scale,))
    if not pairs:
        return {"psnr": 0.0, "ssim": 0.0, "psnr_y": 0.0, "ssim_y": 0.0, "n": 0}
    st = max(1, len(pairs) // max_pairs)
    pairs = pairs[::st][:max_pairs]
    psnrs, ssims, pys, ssys = [], [], [], []
    for lr_path, hr_path, sc in pairs:
        lr_np = load_png(lr_path)
        hr_np = load_png(hr_path)
        h, w = lr_np.shape[:2]
        hr_np = hr_np[: h * sc, : w * sc]
        sr = tiled_forward(model, to_tensor(lr_np), sc, tile=tile)
        hr_t = to_tensor(hr_np)
        hh = min(sr.shape[1], hr_t.shape[1])
        ww = min(sr.shape[2], hr_t.shape[2])
        s, hy = sr[:, :hh, :ww], hr_t[:, :hh, :ww]
        psnrs.append(psnr_torch(s, hy))
        ssims.append(ssim_torch(s, hy))
        ys, yh = rgb_to_y_torch(s), rgb_to_y_torch(hy)
        pys.append(psnr_torch(ys, yh))
        ssys.append(ssim_torch(ys, yh))
    model.train()
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "psnr_y": float(np.mean(pys)),
        "ssim_y": float(np.mean(ssys)),
        "n": len(pairs),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} arch={args.arch} size={args.model_size}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    if args.smoke:
        args.steps = 15
        args.max_train_pairs = 2
        args.batch_size = 2
        args.lr_patch = 48
        args.save_every = 15
        args.eval_every = 0
        cameras = ("Canon",)

    ds = RealSRPairDataset(
        root=args.data_root,
        split="Train",
        cameras=cameras,
        scales=(args.scale,),
        lr_patch=args.lr_patch,
        augment=not args.smoke,
        max_pairs=args.max_train_pairs,
    )
    print(
        f"train pairs: {len(ds)}  scale={args.scale}  LR{args.lr_patch} "
        f"batch={args.batch_size} accum={args.grad_accum}"
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model, tag, n_params = build_any(args)
    model = model.to(device)
    print(f"{tag} params: {n_params:.2f}M  x{args.scale}")

    use_uwcl = (args.arch == "mod" or args.use_uwcl) and not args.l1_only
    from .losses import OffsetAlignedLoss, RealSRLoss

    if args.align_loss:
        criterion = OffsetAlignedLoss(
            max_shift=args.max_shift, w_pure=args.w_pure, w_off=args.w_off
        ).to(device)
        align_nets = [criterion.offset_net]
    else:
        criterion = RealSRLoss(
            w_l1=1.0,
            w_amp_patch=args.w_amp,
            w_hf_conf=args.w_hf_conf,
            w_amp_global=0.0,
            w_grad=args.w_grad if args.w_grad > 0 else 0.0,
            patch=32,
            stride=16,
            conf_win=15,
        ).to(device)
        align_nets = []
    use_criterion = True

    optim = torch.optim.AdamW(
        list(model.parameters()) + [p for n in align_nets for p in n.parameters()],
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and amp_dtype == torch.float16)

    ema_decay = float(args.ema)
    ema_model = None
    if ema_decay > 0:
        import copy

        ema_model = copy.deepcopy(model).eval()
        for p_ in ema_model.parameters():
            p_.requires_grad_(False)

    def ema_update():
        if ema_model is None:
            return
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(ema_decay).add_(msd[k].detach(), alpha=1 - ema_decay)
                else:
                    v.copy_(msd[k])

    best_psnr = -1.0
    nan_streak = 0
    step = 0
    cur_lr_scale = 1.0
    t0 = time.time()
    model.train()
    data_iter = iter(loader)
    meta = {"arch": args.arch, "model_size": args.model_size, "params_m": n_params, "tag": tag}

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt and args.amp and amp_dtype == torch.float16:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        step = int(ckpt.get("step", 0))
        cur_lr_scale = float(ckpt.get("cur_lr_scale", 1.0))
        best_psnr = float(ckpt.get("best_psnr", -1.0))
        if ema_model is not None:
            ema_model.load_state_dict(model.state_dict())
        print(f"resumed from {args.resume} @ step {step} best_psnr={best_psnr:.2f}")

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        cur_lr = lr_at(step, args.lr, args.warmup, args.steps) * cur_lr_scale
        for g in optim.param_groups:
            g["lr"] = cur_lr

        with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
            sr, log_var = model(lr)
        sr = sr.float()
        if log_var is not None:
            log_var = log_var.float()

        if True:  # always criterion
            total, parts = criterion(sr, log_var, hr)
            # ensure logging keys exist
            parts.setdefault("plain", parts.get("l1", total))
            parts.setdefault("uw_l1", parts.get("l1", total))
            parts.setdefault("ctx", torch.zeros((), device=device))
            parts.setdefault("grad", torch.zeros((), device=device))
            parts.setdefault("amp_patch", torch.zeros((), device=device))
            parts.setdefault("hf_conf", torch.zeros((), device=device))

        if not torch.isfinite(total):
            nan_streak += 1
            optim.zero_grad(set_to_none=True)
            if nan_streak >= 20:
                cur_lr_scale = max(cur_lr_scale * args.nan_lr_scale, 0.05)
                print(f"[nan] streak={nan_streak} reduce lr_scale -> {cur_lr_scale:.3f}", flush=True)
                nan_streak = 0
            step += 1
            continue
        nan_streak = 0

        scaler.scale(total / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            ema_update()

        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                cur_psnr = psnr(sr, hr)
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
            msg = (
                f"step {step:05d}/{args.steps} | loss {float(total):.4f} "
                f"(l1 {float(parts.get('l1', 0)):.4f} aln {float(parts.get('l1_aln', 0)):.4f} "
                f"ampP {float(parts.get('amp_patch', 0)):.4f} hfC {float(parts.get('hf_conf', 0)):.4f}) "
                f"| psnr {float(cur_psnr):.2f} "
                f"| lr {cur_lr:.2e} | peakVRAM {mem:.2f}GB | {time.time()-t0:.0f}s"
            )
            print(msg, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(total),
                            "plain": float(parts["plain"]),
                            "train_psnr": float(cur_psnr),
                            "lr": cur_lr,
                            "vram_gb": mem,
                            "lr_scale": cur_lr_scale,
                        }
                    )
                    + "\n"
                )

        do_eval = args.eval_every > 0 and ((step + 1) % args.eval_every == 0 or step == args.steps - 1)
        do_save = (step + 1) % args.save_every == 0 or step == args.steps - 1 or do_eval

        if do_eval:
            eval_net = ema_model if ema_model is not None else model
            ev = run_eval(eval_net, args.data_root, args.scale, cameras, args.eval_pairs, args.eval_tile)
            print(
                f"  >> EVAL step {step+1}: RGB {ev['psnr']:.2f}/{ev['ssim']:.4f}  "
                f"Y {ev.get('psnr_y', 0):.2f}/{ev.get('ssim_y', 0):.4f}  (n={ev['n']})",
                flush=True,
            )
            with (out_dir / "eval_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step + 1, **ev}) + "\n")
            if ev["psnr"] > best_psnr:
                best_psnr = ev["psnr"]
                save_sd = eval_net.state_dict()
                torch.save(
                    {
                        "step": step + 1,
                        "model": save_sd,
                        "optim": optim.state_dict(),
                        "args": vars(args),
                        "meta": meta,
                        "best_psnr": best_psnr,
                        "cur_lr_scale": cur_lr_scale,
                        "ema": ema_decay > 0,
                    },
                    out_dir / "ckpt_best.pt",
                )
                print(f"  >> new best PSNR {best_psnr:.2f} -> ckpt_best.pt", flush=True)

        if do_save:
            save_sd = (ema_model if ema_model is not None else model).state_dict()
            torch.save(
                {
                    "step": step + 1,
                    "model": save_sd,
                    "optim": optim.state_dict(),
                    "args": vars(args),
                    "meta": meta,
                    "best_psnr": best_psnr,
                    "cur_lr_scale": cur_lr_scale,
                    "ema": ema_decay > 0,
                },
                out_dir / "ckpt_last.pt",
            )
            if (step + 1) % args.save_every == 0:
                torch.save(
                    {
                        "step": step + 1,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "args": vars(args),
                        "meta": meta,
                    },
                    out_dir / f"ckpt_step{step+1}.pt",
                )
            print(f"  saved ckpt @ step {step+1}", flush=True)

        step += 1

    if device.type == "cuda":
        print(f"done. peak VRAM {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
    print(f"best eval PSNR: {best_psnr:.2f} dB")
    (out_dir / "summary.json").write_text(
        json.dumps({**meta, "best_psnr": best_psnr, "steps": args.steps, "args": vars(args)}, indent=2),
        encoding="utf-8",
    )
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
