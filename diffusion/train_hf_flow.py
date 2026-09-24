"""Train the residual HF rectified-flow head on top of a frozen regression SR.

Direction C.  Loads a frozen regression model (the champion b64 + wavelet-HF loss),
runs it to get `pred`, and trains hf_flow.build_hf_head to generate the *high-frequency
residual* in Haar subbands (conditioned on the regression's LL + HF mean).  Because the
head only edits HF, the regression's fidelity (LL) is preserved -- we chase MUSIQ/MANIQA
without the full-image-flow PSNR collapse.

  python -m diffusion.train_hf_flow \
    --reg-ckpt experiments/diffusion/stack_dwt/b64_pre_dwt8/ckpt_best.pt \
    --out experiments/diffusion/hf_flow/hf_x3s1 --steps 8000 --batch 64
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
from torch.utils.data import DataLoader, RandomSampler

from .data import RealSRCropDataset, make_split, default_root
from .hf_flow import build_hf_head, hf_flow_training_loss
from .unet import build_unet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="")
    p.add_argument("--reg-ckpt", required=True, help="frozen regression SR (b64+dwt champion)")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--lr-patch", type=int, default=64)
    p.add_argument("--head-base", type=int, default=64, help="HF flow UNet base")
    p.add_argument("--head-mult", default="1,2,4", help="HF flow UNet channel mult (half-res)")
    p.add_argument("--head-res", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-pairs", type=int, default=16)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--decoded-manifest", default="data/decoded/manifest.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if not a.data_root:
        a.data_root = default_root()
    return a


def load_frozen_regression(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ck.get("args", {}) or {}
    ns = int(targs.get("native_lr", 0))
    R = build_unet(targs.get("size", "S"), input_size=128, in_channels=3,
                   base=int(targs.get("base", 64)),
                   mult=tuple(int(x) for x in str(targs.get("mult", "1,2,4,4")).split(",")),
                   num_res=int(targs.get("num_res", 2)),
                   attn_levels=tuple(int(x) for x in str(targs.get("attn_levels", "2,3")).split(",")),
                   in_stride=2 if ns else 1, out_scale=2 if ns else 1).to(device)
    R.load_state_dict(ck["model"])
    R.eval()
    for q in R.parameters():
        q.requires_grad_(False)
    return R


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def _wi(_):
    torch.set_num_threads(1)


@torch.no_grad()
def val_flow_loss(head, R, val_ds, device):
    head.eval()
    tot, n = 0.0, 0
    t0 = torch.zeros(1, device=device)
    for i in range(min(len(val_ds), 64)):
        b = val_ds[i]
        lr = b["lr"].unsqueeze(0).to(device)
        hr = b["hr"].unsqueeze(0).to(device)
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        pred = (lr_up + R(lr_up, t0)).clamp(0, 1)
        loss, _ = hf_flow_training_loss(head, pred, hr)
        tot += float(loss); n += 1
    head.train()
    return tot / max(n, 1)


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")

    R = load_frozen_regression(a.reg_ckpt, device)
    ds, val_ds = make_split(RealSRCropDataset(a.data_root, "Train", ("Canon", "Nikon"),
                                              a.scale, a.lr_patch, True, cache=False,
                                              decoded_manifest=a.decoded_manifest or None), a.val_pairs, a.seed)
    sampler = RandomSampler(ds, replacement=True, num_samples=a.batch * 8) if len(ds) <= a.batch else None
    loader = DataLoader(ds, batch_size=a.batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=a.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=a.num_workers > 0, worker_init_fn=_wi if a.num_workers > 0 else None)
    head = build_hf_head("S", base=a.head_base, ch=3, mult=tuple(int(x) for x in a.head_mult.split(",")),
                         num_res=a.head_res, attn_levels=()).to(device)
    print(f"hf head: {sum(p.numel() for p in head.parameters())/1e6:.2f}M  reg frozen "
          f"{a.reg_ckpt}  train={len(ds)} val={0 if val_ds is None else len(val_ds)}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp))
    ema = copy.deepcopy(head).eval() if a.ema > 0 else None
    if ema is not None:
        for q in ema.parameters():
            q.requires_grad_(False)
    t_zeros = torch.zeros(1, device=device)

    def ema_update():
        if ema is None:
            return
        with torch.no_grad():
            for x, y in zip(ema.parameters(), head.parameters()):
                x.mul_(a.ema).add_(y.detach(), alpha=1 - a.ema)

    def payload(step, best):
        return {"step": step, "model": (ema or head).state_dict(),
                "head_raw": head.state_dict(), "best": best, "args": vars(a),
                "reg_ckpt": a.reg_ckpt, "scale": a.scale}

    step, best = 0, float("inf")
    t0 = time.time()
    it = iter(loader)
    head.train()
    while step < a.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        with torch.no_grad():
            lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
            pred = (lr_up + R(lr_up, t_zeros.expand(lr_up.shape[0]))).clamp(0, 1)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.lr, a.warmup, a.steps)
        with torch.amp.autocast("cuda", enabled=bool(a.amp), dtype=torch.bfloat16):
            loss, _ = hf_flow_training_loss(head, pred, hr)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); step += 1; continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        scaler.step(opt); scaler.update(); ema_update()
        if step % 50 == 0:
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
            print(f"step {step:06d}/{a.steps} | flowL {float(loss.detach()):.4f} | "
                  f"{mem:.1f}GB | {time.time()-t0:.0f}s", flush=True)
        if a.eval_every > 0 and (step + 1) % a.eval_every == 0 and val_ds is not None:
            vl = val_flow_loss(head, R, val_ds, device)
            print(f"  VALFLOW {step+1}: {vl:.5f}", flush=True)
            if vl < best:
                best = vl
                torch.save(payload(step + 1, best), out / "ckpt_best.pt")
        if (step + 1) % a.save_every == 0:
            torch.save(payload(step + 1, best), out / "ckpt_last.pt")
        step += 1
    torch.save(payload(step, best), out / "ckpt_last.pt")
    print(f"done best_val_flow={best:.5f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
