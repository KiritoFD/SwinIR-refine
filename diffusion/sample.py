"""Save side-by-side LR-up / SR / HR PNGs from a DiT ckpt (quick visual check)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import build_index
from .data import default_root
from .dit import build_dit
from .eval_official import sr_latent_tiled, sr_pixel_tiled
from .vae import load_vae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="", help="auto-detected if empty")
    p.add_argument("--vae", default="")
    p.add_argument("--mode", default="auto", choices=["auto", "latent", "pixel"])
    p.add_argument("--objective", default="auto", choices=["auto", "flow", "reg"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--tile", type=int, default=0, help="0 = training crop")
    p.add_argument("--pad", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not getattr(args, "data_root", ""):
        args.data_root = default_root()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ck.get("args", {}) or {}
    mode = args.mode
    if mode == "auto":
        mode = ck.get("mode") or ("latent" if ("latent_size" in ck or targs.get("vae")) else "pixel")
    objective = args.objective
    if objective == "auto":
        objective = ck.get("objective") or targs.get("objective") or "flow"
    in_ch = int(ck.get("in_channels") or targs.get("in_channels") or (32 if mode == "latent" else 6))
    size = targs.get("size", "S")
    patch = int(targs.get("patch", 2))
    residual = bool(targs.get("residual", False))

    vae = vinfo = None
    tile = args.tile
    if mode == "latent":
        vae_name = args.vae or ck.get("vae") or targs.get("vae") or "flux1-vae"
        vae, vinfo = load_vae(vae_name, device=device)
        input_size = int(ck.get("latent_size") or (targs.get("lr_patch", 128) * targs.get("scale", 2) // vinfo.downscale))
        if tile <= 0:
            tile = int(targs.get("lr_patch", 128))
    else:
        input_size = int(ck.get("hr_px") or targs.get("lr_patch", 64) * targs.get("scale", 2))
        if tile <= 0:
            tile = input_size // int(targs.get("scale", 2))

    model = build_dit(size, input_size=input_size, patch_size=patch, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    pairs = build_index(args.data_root, ("Canon", "Nikon"), "Test", 2)
    step = max(1, len(pairs) // max(args.n, 1))
    pairs = pairs[::step][: args.n]
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "samples"
    out.mkdir(parents=True, exist_ok=True)

    for lr_path, hr_path, sc in pairs:
        lr = np.asarray(Image.open(lr_path).convert("RGB"), dtype=np.uint8)
        hr = np.asarray(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
        if mode == "latent":
            sr = sr_latent_tiled(model, vae, vinfo, lr, sc, args.steps, tile, args.pad, device, objective, args.seed)
        else:
            sr = sr_pixel_tiled(model, lr, sc, args.steps, tile, args.pad, residual, device, objective, args.seed)
        h, w = lr.shape[:2]
        up = np.asarray(Image.fromarray(lr).resize((w * sc, h * sc), Image.Resampling.BICUBIC))
        hh = min(sr.shape[0], hr.shape[0], up.shape[0])
        ww = min(sr.shape[1], hr.shape[1], up.shape[1])
        canvas = np.concatenate([up[:hh, :ww], sr[:hh, :ww], hr[:hh, :ww]], axis=1)
        name = Path(lr_path).stem
        Image.fromarray(canvas).save(out / f"{name}_cmp.png")
        print("wrote", out / f"{name}_cmp.png", flush=True)


if __name__ == "__main__":
    main()
