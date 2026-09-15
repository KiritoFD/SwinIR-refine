"""Official RealSR V3 Test evaluation (Cai et al. ICCV 2019 Test.m).

Protocol (paper + official README + Test.m):
  - Train RGB, evaluate Y channel only (primary metric).
  - MATLAB rgb2ycbcr limited-range Y → im2uint8 → PSNR/SSIM, crop=0.
  - modcrop both LR and HR by 4.
  - Full Canon+Nikon Test (×2: 50+50=100 pairs).
  - Auxiliary RGB PSNR kept for internal monitoring only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_realsr_index
from .metrics import modcrop, official_pair_metrics
from .model import build_model


def load_rgb_u8(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = np.ascontiguousarray(img)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def tiled_forward(model, lr: torch.Tensor, scale: int, tile: int = 256, pad: int = 16):
    """lr: (3,H,W) float [0,1]. Overlapping tiles, average in valid region."""
    _, H, W = lr.shape
    device = next(model.parameters()).device
    out_h, out_w = H * scale, W * scale
    # Accumulate on CPU: 2k+ SR tensors on an 8GB card OOM under RAPE FiLM.
    sr = torch.zeros(3, out_h, out_w)
    acc = torch.zeros(1, out_h, out_w)

    ys = list(range(0, max(H - tile, 0) + 1, tile))
    if not ys or ys[-1] + tile < H:
        ys.append(max(H - tile, 0))
    xs = list(range(0, max(W - tile, 0) + 1, tile))
    if not xs or xs[-1] + tile < W:
        xs.append(max(W - tile, 0))
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    for y in ys:
        for x in xs:
            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y1 = min(H, y + tile + pad)
            x1 = min(W, x + tile + pad)
            patch = lr[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            out, _ = model(patch)
            out = out[0].clamp(0, 1).float().cpu()
            del patch
            ry0 = (y - y0) * scale
            rx0 = (x - x0) * scale
            ry1 = ry0 + min(tile, H - y) * scale
            rx1 = rx0 + min(tile, W - x) * scale
            dst = (
                slice(None),
                slice(ry0 + y0 * scale, ry1 + y0 * scale),
                slice(rx0 + x0 * scale, rx1 + x0 * scale),
            )
            src = (slice(None), slice(ry0, ry1), slice(rx0, rx1))
            sr[dst] += out[src]
            acc[dst] += 1.0
            del out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sr = sr / acc.clamp(min=1.0)
    return (sr.clamp(0, 1) * 255.0).round().byte().cpu().permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=r"G:\RealSR\experiments\improve\E11_align_loss\ckpt_best.pt")
    p.add_argument("--data-root", type=str, default=r"G:\RealSR\data\RealSR(V3)")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--cameras", type=str, default="Canon,Nikon")
    p.add_argument("--max-pairs", type=int, default=0, help="0 = full official Test set")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-images", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    meta = ckpt.get("meta", {})
    upscale = train_args.get("scale", args.scale)
    model_size = train_args.get("model_size", "base")
    arch = train_args.get("arch", meta.get("arch", "mod"))

    if arch == "swinir":
        from .swinir_baseline import SwinIRTrainWrapper, build_swinir

        model = SwinIRTrainWrapper(build_swinir(size=model_size, upscale=upscale)).to(device)
    else:
        model = build_model(upscale=upscale, size=model_size).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"ignore extra ckpt keys ({len(unexpected)}): {unexpected[:6]}")
    model.eval()
    n_params = sum(q.numel() for q in model.parameters()) / 1e6
    print(f"loaded {ckpt_path} step={ckpt.get('step')} {arch}/{model_size} ({n_params:.2f}M) x{upscale}")

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pairs = build_realsr_index(args.data_root, cameras, "Test", (args.scale,))
    if args.max_pairs and args.max_pairs > 0:
        step = max(1, len(pairs) // args.max_pairs)
        pairs = pairs[::step][: args.max_pairs]
    print(f"official eval pairs: {len(pairs)} (cameras={cameras}, scale={args.scale})")

    out_dir = Path(args.out) if args.out else ckpt_path.parent / "eval_official"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        lr_u8 = load_rgb_u8(lr_path)
        hr_u8 = load_rgb_u8(hr_path)
        lr_u8 = modcrop(lr_u8, 4)
        hr_u8 = modcrop(hr_u8, 4)
        h, w = lr_u8.shape[:2]
        hr_u8 = hr_u8[: h * sc, : w * sc]
        lr_t = to_tensor(lr_u8)
        sr_u8 = tiled_forward(model, lr_t, sc, tile=args.tile)
        hh = min(sr_u8.shape[0], hr_u8.shape[0])
        ww = min(sr_u8.shape[1], hr_u8.shape[1])
        m = official_pair_metrics(sr_u8[:hh, :ww], hr_u8[:hh, :ww])
        name = Path(lr_path).name
        m.update({"name": name, "scale": sc, "lr": lr_path})
        rows.append(m)
        print(
            f"[{i+1}/{len(pairs)}] {name} x{sc}  "
            f"Y {m['psnr_y']:.2f}/{m['ssim_y']:.4f}  RGB {m['psnr_rgb']:.2f}",
            flush=True,
        )
        if args.save_images:
            lr_up = np.asarray(
                Image.fromarray(lr_u8).resize((w * sc, h * sc), Image.Resampling.BICUBIC)
            )
            canvas = np.concatenate([lr_up, sr_u8, hr_u8], axis=1)
            Image.fromarray(canvas).save(out_dir / f"{Path(name).stem}_x{sc}_cmp.png")

    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    summary = {
        "protocol": "RealSR official Test.m (Y limited-range uint8, crop=0, modcrop=4)",
        "paper": "Cai et al., Toward Real-World Single Image Super-Resolution, ICCV 2019",
        "ckpt": str(ckpt_path),
        "scale": args.scale,
        "n": len(rows),
        "psnr_y": mean("psnr_y"),
        "ssim_y": mean("ssim_y"),
        "psnr_rgb": mean("psnr_rgb"),
        "ssim_rgb": mean("ssim_rgb"),
        "per_image": rows,
    }
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n== OFFICIAL n={len(rows)} | "
        f"Y {summary['psnr_y']:.4f}/{summary['ssim_y']:.4f} | "
        f"RGB(aux) {summary['psnr_rgb']:.2f}/{summary['ssim_rgb']:.4f} =="
    )
    print(f"wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
