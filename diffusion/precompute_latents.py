"""Encode every Train pair into VAE latents once, so training is not VAE-bound.

Measured on the 4090: encoding 2x HR512 crops per step costs ~1.0 s/step while
the DiT itself is ~0.17 s/step. Pre-encoding removes ~85% of the latent step
time and lets us afford HR512 crops + more steps + more VAEs.

  python -m diffusion.precompute_latents --vae flux1-vae --out data/latents/flux1-vae
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import build_index
from .data import default_root
from .vae import encode, load_vae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="", help="auto-detected if empty")
    p.add_argument("--out", required=True, help="cache dir, e.g. data/latents/flux1-vae")
    p.add_argument("--vae", default="flux1-vae")
    p.add_argument("--vae-dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--split", default="Train")
    p.add_argument("--cameras", default="Canon,Nikon")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", type=int, default=1)
    args = p.parse_args()
    if not getattr(args, "data_root", ""):
        args.data_root = default_root()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.vae_dtype == "bf16" else torch.float32
    vae, vinfo = load_vae(args.vae, device=device, dtype=dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(
        json.dumps({"vae": args.vae, **{k: getattr(vinfo, k) for k in
                    ("downscale", "latent_channels", "scaling_factor", "shift_factor")}}, indent=2),
        encoding="utf-8",
    )

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pairs = build_index(args.data_root, cameras, args.split, args.scale)
    print(f"{len(pairs)} pairs, VAE {vinfo.name} {vinfo.norm_tag}", flush=True)

    ds = vinfo.downscale
    t0 = time.time()
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        stem = f"{Path(lr_path).parent.parent.name}_{Path(lr_path).stem}"
        dst = out / f"{stem}.pt"
        if args.skip_existing and dst.is_file():
            continue
        lr = np.asarray(Image.open(lr_path).convert("RGB"), dtype=np.uint8)
        hr = np.asarray(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
        th, tw = hr.shape[0], hr.shape[1]
        # keep everything on the exact 8x grid so encode/decode round-trips
        th -= th % (ds * sc)
        tw -= tw % (ds * sc)
        # crop LR to match, then bicubic-up: keeps the pair exactly aligned
        lr = lr[: th // sc, : tw // sc]
        hr_t = torch.from_numpy(np.ascontiguousarray(hr[:th, :tw])).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        lr_t = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        import torch.nn.functional as F

        lr_up = F.interpolate(lr_t, size=(th, tw), mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.no_grad():
            z_hr = encode(vae, hr_t.to(device), vinfo, sample=False)[0].half().cpu()
            z_lr = encode(vae, lr_up.to(device), vinfo, sample=False)[0].half().cpu()
        torch.save({"z_hr": z_hr, "z_lr": z_lr, "hr_size": (th, tw), "name": stem}, dst)
        if i % 25 == 0 or i == len(pairs) - 1:
            print(f"  {i+1}/{len(pairs)} {stem} {tuple(z_hr.shape)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(pairs)} pairs -> {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
