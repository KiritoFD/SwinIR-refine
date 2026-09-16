"""Official full-set RealSR eval for a trained latent/pixel DiT ckpt.

Tiled ODE sampling on full Test images + official Y metrics.

Example:
  python -m diffusion.eval_official \
    --ckpt experiments/diffusion/latent_dit_flux1-dev/ckpt_best.pt \
    --vae flux1-dev --max-pairs 0 --steps 20 --tile 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_index
from .dit import build_dit
from .flow import sample_flow
from .metrics import official_pair_metrics
from .vae import decode, encode, load_vae


def load_rgb_u8(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_tensor(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float() / 255.0


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def tile_coords(h: int, w: int, tile: int):
    ys = list(range(0, max(h - tile, 0) + 1, tile))
    xs = list(range(0, max(w - tile, 0) + 1, tile))
    if not ys or ys[-1] + tile < h:
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] + tile < w:
        xs.append(max(w - tile, 0))
    return sorted(set(ys)), sorted(set(xs))


@torch.no_grad()
def sr_latent_tiled(model, vae, vinfo, lr_u8: np.ndarray, scale: int, steps: int, tile: int, pad: int, device):
    """Bicubic-up LR → tile encode cond → flow sample latent tiles → decode → stitch."""
    h, w = lr_u8.shape[:2]
    hr_h, hr_w = h * scale, w * scale
    ds = vinfo.downscale
    # pad HR canvas to multiple of ds
    ph = (ds - hr_h % ds) % ds
    pw = (ds - hr_w % ds) % ds
    out = torch.zeros(1, 3, hr_h + ph, hr_w + pw, device=device)
    acc = torch.zeros(1, 1, hr_h + ph, hr_w + pw, device=device)

    lr_t = to_tensor(lr_u8).unsqueeze(0).to(device)
    lr_up = F.interpolate(lr_t, size=(hr_h + ph, hr_w + pw), mode="bicubic", align_corners=False).clamp(0, 1)

    # work in latent grid
    # sample in HR-pixel tiles aligned to ds
    tsize = tile * scale  # HR tile
    tsize = (tsize // ds) * ds
    if tsize < ds * 8:
        tsize = ds * 8
    ys, xs = tile_coords(hr_h + ph, hr_w + pw, tsize)
    for y in ys:
        for x in xs:
            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y1 = min(hr_h + ph, y + tsize + pad)
            x1 = min(hr_w + pw, x + tsize + pad)
            y0 -= y0 % ds
            x0 -= x0 % ds
            y1 = min(hr_h + ph, y1 + (ds - y1 % ds) % ds)
            x1 = min(hr_w + pw, x1 + (ds - x1 % ds) % ds)
            patch = lr_up[:, :, y0:y1, x0:x1]
            zc = encode(vae, patch, vinfo, sample=False)
            lh, lw = zc.shape[-2:]
            z = sample_flow(
                model,
                (1, vinfo.latent_channels, lh, lw),
                cond=zc,
                steps=steps,
                solver="heun",
                device=device,
            )
            rec = decode(vae, z, vinfo)
            ry0 = y - y0
            rx0 = x - x0
            ry1 = ry0 + min(tsize, hr_h + ph - y)
            rx1 = rx0 + min(tsize, hr_w + pw - x)
            out[:, :, y0 + ry0 : y0 + ry1, x0 + rx0 : x0 + rx1] += rec[:, :, ry0:ry1, rx0:rx1]
            acc[:, :, y0 + ry0 : y0 + ry1, x0 + rx0 : x0 + rx1] += 1.0
            del zc, z, rec, patch
            if device.type == "cuda":
                torch.cuda.empty_cache()
    out = out / acc.clamp(min=1.0)
    return to_u8(out[0, :, :hr_h, :hr_w])


@torch.no_grad()
def sr_pixel_tiled(model, lr_u8: np.ndarray, scale: int, steps: int, tile: int, pad: int, residual: bool, device):
    h, w = lr_u8.shape[:2]
    hr_h, hr_w = h * scale, w * scale
    out = torch.zeros(1, 3, hr_h, hr_w, device=device)
    acc = torch.zeros(1, 1, hr_h, hr_w, device=device)
    lr_t = to_tensor(lr_u8).unsqueeze(0).to(device)
    lr_up = F.interpolate(lr_t, size=(hr_h, hr_w), mode="bicubic", align_corners=False).clamp(0, 1)
    tsize = tile * scale
    ys, xs = tile_coords(hr_h, hr_w, tsize)
    for y in ys:
        for x in xs:
            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y1 = min(hr_h, y + tsize + pad)
            x1 = min(hr_w, x + tsize + pad)
            # align to model patch
            ps = 2
            y0 -= y0 % ps
            x0 -= x0 % ps
            y1 = min(hr_h, y1 + (ps - y1 % ps) % ps)
            x1 = min(hr_w, x1 + (ps - x1 % ps) % ps)
            cond = lr_up[:, :, y0:y1, x0:x1]
            ch, cw = cond.shape[-2:]
            z = sample_flow(
                model,
                (1, 3, ch, cw),
                cond=cond,
                steps=steps,
                solver="heun",
                device=device,
            )
            rec = (z * 2.0 + cond).clamp(0, 1) if residual else z.clamp(0, 1)
            ry0 = y - y0
            rx0 = x - x0
            ry1 = ry0 + min(tsize, hr_h - y)
            rx1 = rx0 + min(tsize, hr_w - x)
            out[:, :, y0 + ry0 : y0 + ry1, x0 + rx0 : x0 + rx1] += rec[:, :, ry0:ry1, rx0:rx1]
            acc[:, :, y0 + ry0 : y0 + ry1, x0 + rx0 : x0 + rx1] += 1.0
            del z, rec, cond
            if device.type == "cuda":
                torch.cuda.empty_cache()
    out = out / acc.clamp(min=1.0)
    return to_u8(out[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default=r"G:\RealSR\data\RealSR(V3)")
    p.add_argument("--vae", default="", help="required for latent mode")
    p.add_argument("--mode", default="auto", choices=["auto", "latent", "pixel"])
    p.add_argument("--cameras", default="Canon,Nikon")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--tile", type=int, default=128, help="LR tile size")
    p.add_argument("--pad", type=int, default=16)
    p.add_argument("--out", default="")
    p.add_argument("--residual", action="store_true", help="pixel mode: HR = residual + bicubic-up")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ck.get("args", {})
    mode = args.mode
    if mode == "auto":
        mode = "latent" if "latent_size" in ck or targs.get("vae") else "pixel"

    in_ch = int(ck.get("in_channels") or targs.get("in_channels") or (8 if mode == "latent" else 6))
    size = targs.get("size", "S")
    patch = int(targs.get("patch", 2))
    residual = bool(getattr(args, "residual", False) or targs.get("residual", False))

    vae = vinfo = None
    if mode == "latent":
        vae_name = args.vae or ck.get("vae") or targs.get("vae") or "flux1-dev"
        vae, vinfo = load_vae(vae_name, device=device)
        lat = int(ck.get("latent_size") or (targs.get("lr_patch", 64) * targs.get("scale", 2) // vinfo.downscale))
        input_size = lat
    else:
        hr_px = int(ck.get("hr_px") or targs.get("lr_patch", 64) * targs.get("scale", 2))
        input_size = hr_px

    model = build_dit(size, input_size=input_size, patch_size=patch, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path} mode={mode} in_ch={in_ch} size={size} input={input_size}", flush=True)

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pairs = build_index(args.data_root, cameras, "Test", args.scale)
    if args.max_pairs and args.max_pairs > 0:
        step = max(1, len(pairs) // args.max_pairs)
        pairs = pairs[::step][: args.max_pairs]
    print(f"pairs: {len(pairs)}", flush=True)

    out_dir = Path(args.out) if args.out else ckpt_path.parent / "eval_official"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        lr_u8 = load_rgb_u8(lr_path)
        hr_u8 = load_rgb_u8(hr_path)
        try:
            if mode == "latent":
                sr = sr_latent_tiled(model, vae, vinfo, lr_u8, sc, args.steps, args.tile, args.pad, device)
            else:
                sr = sr_pixel_tiled(model, lr_u8, sc, args.steps, args.tile, args.pad, residual, device)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(pairs)}] FAIL {Path(lr_path).name}: {e}", flush=True)
            continue
        hh = min(sr.shape[0], hr_u8.shape[0])
        ww = min(sr.shape[1], hr_u8.shape[1])
        m = official_pair_metrics(sr[:hh, :ww], hr_u8[:hh, :ww])
        m["name"] = Path(lr_path).name
        rows.append(m)
        print(
            f"[{i+1}/{len(pairs)}] {m['name']} Y {m['psnr_y']:.2f}/{m['ssim_y']:.4f} RGB {m['psnr_rgb']:.2f}",
            flush=True,
        )

    def mean(k):
        vals = [r[k] for r in rows if r.get(k) == r.get(k)]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "protocol": "RealSR official Test.m (Y limited-range uint8, crop=0, modcrop implied by pair crop)",
        "ckpt": str(ckpt_path),
        "mode": mode,
        "n": len(rows),
        "psnr_y": mean("psnr_y"),
        "ssim_y": mean("ssim_y"),
        "psnr_rgb": mean("psnr_rgb"),
        "steps": args.steps,
        "per_image": rows,
    }
    path = out_dir / "eval.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n== OFFICIAL n={summary['n']} | Y {summary['psnr_y']:.4f}/{summary['ssim_y']:.4f} | "
        f"RGB {summary['psnr_rgb']:.2f} ==",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
