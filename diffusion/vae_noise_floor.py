"""VAE reconstruction noise floor on RealSR (full Test pairs or crops).

This bounds how good any latent diffusion can ever be: encode(HR)→decode
must stay close to HR, otherwise Y-PSNR of generated samples is capped.

Usage (server):
  python -m diffusion.vae_noise_floor --vae flux1-dev --max-pairs 30 --tile 256
  python -m diffusion.vae_noise_floor --vae /path/to/local/vae --max-pairs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.vae import decode, encode, load_vae, psnr01, ssim01  # noqa: E402
from model.metrics import official_pair_metrics  # noqa: E402


def to_tensor(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float() / 255.0


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def recon_tile(vae, info, img_u8: np.ndarray, tile: int = 256, pad: int = 16, device="cuda"):
    """Encode/decode tiled so large RealSR Test images fit in VRAM."""
    h, w = img_u8.shape[:2]
    ds = info.downscale
    # pad to multiple of ds
    ph = (ds - h % ds) % ds
    pw = (ds - w % ds) % ds
    if ph or pw:
        img_u8 = np.pad(img_u8, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    hh, ww = img_u8.shape[:2]
    x = to_tensor(img_u8).unsqueeze(0).to(device)
    if max(hh, ww) <= tile * 2:
        z = encode(vae, x, info, sample=False)
        y = decode(vae, z, info)[0]
        return to_u8(y)[:h, :w]

    out = torch.zeros(1, 3, hh, ww, device=device)
    acc = torch.zeros(1, 1, hh, ww, device=device)
    ys = list(range(0, max(hh - tile, 0) + 1, tile))
    xs = list(range(0, max(ww - tile, 0) + 1, tile))
    if not ys or ys[-1] + tile < hh:
        ys.append(max(hh - tile, 0))
    if not xs or xs[-1] + tile < ww:
        xs.append(max(ww - tile, 0))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    for y in ys:
        for x0 in xs:
            y0 = max(0, y - pad)
            x00 = max(0, x0 - pad)
            y1 = min(hh, y + tile + pad)
            x1 = min(ww, x0 + tile + pad)
            # align to ds
            y0 = y0 - (y0 % ds)
            x00 = x00 - (x00 % ds)
            y1 = min(hh, y1 + (ds - y1 % ds) % ds)
            x1 = min(ww, x1 + (ds - x1 % ds) % ds)
            patch = x[:, :, y0:y1, x00:x1]
            z = encode(vae, patch, info, sample=False)
            rec = decode(vae, z, info)
            # paste valid region
            ry0 = (y - y0)
            rx0 = (x0 - x00)
            ry1 = ry0 + min(tile, hh - y)
            rx1 = rx0 + min(tile, ww - x0)
            out[:, :, y0 + ry0 : y0 + ry1, x00 + rx0 : x00 + rx1] += rec[:, :, ry0:ry1, rx0:rx1]
            acc[:, :, y0 + ry0 : y0 + ry1, x00 + rx0 : x00 + rx1] += 1.0
    out = out / acc.clamp(min=1.0)
    return to_u8(out[0])[:h, :w]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"G:\RealSR\data\RealSR(V3)")
    p.add_argument("--vae", default="flux1-dev", help="preset name or local/HF path")
    p.add_argument("--cameras", default="Canon,Nikon")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=30, help="0 = full Test (100)")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--crop", type=int, default=0, help="if >0, eval random crop size instead of full image")
    p.add_argument("--out", default=r"G:\RealSR\experiments\diffusion\vae_noise")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading VAE {args.vae} ...", flush=True)
    vae, info = load_vae(args.vae, device=device, dtype=torch.float32)
    print(
        f"VAE={info.name} src={info.source} downscale={info.downscale} "
        f"C={info.latent_channels} sf={info.scaling_factor}",
        flush=True,
    )

    from diffusion.data import build_index

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pairs = build_index(args.data_root, cameras, "Test", args.scale)
    if args.max_pairs and args.max_pairs > 0:
        step = max(1, len(pairs) // args.max_pairs)
        pairs = pairs[::step][: args.max_pairs]
    print(f"pairs: {len(pairs)}", flush=True)

    rows = []
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        hr = np.asarray(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
        if args.crop and args.crop > 0:
            ch, cw = args.crop, args.crop
            if hr.shape[0] >= ch and hr.shape[1] >= cw:
                top = (hr.shape[0] - ch) // 2
                left = (hr.shape[1] - cw) // 2
                hr = hr[top : top + ch, left : left + cw]
        try:
            rec = recon_tile(vae, info, hr, tile=args.tile, device=device)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(pairs)}] FAIL {Path(hr_path).name}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        m = official_pair_metrics(rec, hr[: rec.shape[0], : rec.shape[1]])
        rec_t = to_tensor(rec).unsqueeze(0).to(device)
        hr_t = to_tensor(hr[: rec.shape[0], : rec.shape[1]]).unsqueeze(0).to(device)
        m["psnr01"] = psnr01(rec_t, hr_t)
        m["ssim01"] = ssim01(rec_t, hr_t)
        m["name"] = Path(hr_path).name
        m["h"], m["w"] = hr.shape[:2]
        rows.append(m)
        print(
            f"[{i+1}/{len(pairs)}] {m['name']} {hr.shape[0]}x{hr.shape[1]} "
            f"Y {m['psnr_y']:.2f}/{m['ssim_y']:.4f} RGB01 {m['psnr01']:.2f}/{m['ssim01']:.4f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def mean(k):
        vals = [r[k] for r in rows if r.get(k) == r.get(k)]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "vae": info.name,
        "source": info.source,
        "downscale": info.downscale,
        "latent_channels": info.latent_channels,
        "scaling_factor": info.scaling_factor,
        "n": len(rows),
        "psnr_y": mean("psnr_y"),
        "ssim_y": mean("ssim_y"),
        "psnr_rgb": mean("psnr_rgb"),
        "psnr01": mean("psnr01"),
        "ssim01": mean("ssim01"),
        "per_image": rows,
    }
    safe = info.name.replace("/", "_").replace("\\", "_")
    path = out_dir / f"noise_floor_{safe}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n== VAE NOISE FLOOR ==", flush=True)
    print(
        f"n={summary['n']}  Y {summary['psnr_y']:.4f}/{summary['ssim_y']:.4f}  "
        f"RGB01 {summary['psnr01']:.4f}/{summary['ssim01']:.4f}",
        flush=True,
    )
    print(f"wrote {path}", flush=True)
    print(
        "Interpretation: any latent DiT is capped near this Y-PSNR "
        "(decode errors cannot be recovered by the generative model).",
        flush=True,
    )


if __name__ == "__main__":
    main()
