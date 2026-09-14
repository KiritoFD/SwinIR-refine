"""Tiled evaluation on RealSR V3 Test set (PSNR/SSIM)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_realsr_index
from .model import build_model


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def load_png(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr_torch(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(sr.clamp(0, 1), hr.clamp(0, 1)).item()
    if mse <= 1e-12:
        return 99.0
    return 10 * math.log10(max_val * max_val / mse)


def rgb_to_y_torch(img: torch.Tensor) -> torch.Tensor:
    """BT.601 Y, accepts (3,H,W) or (B,3,H,W)."""
    if img.dim() == 3:
        r, g, b = img[0], img[1], img[2]
        return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(0)
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def ssim_torch(sr: torch.Tensor, hr: torch.Tensor, window: int = 11) -> float:
    """Standard SSIM on CHW/BCHW RGB, gaussian window."""
    if sr.dim() == 3:
        sr = sr.unsqueeze(0)
        hr = hr.unsqueeze(0)
    C = sr.shape[1]
    gauss = torch.arange(window, device=sr.device, dtype=sr.dtype) - window // 2
    gauss = torch.exp(-(gauss**2) / 2.0)
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    kernel_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    kernel = kernel_2d.expand(C, 1, window, window).contiguous()
    pad = window // 2

    def filt(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_s, mu_h = filt(sr), filt(hr)
    mu_s2, mu_h2, mu_sh = mu_s * mu_s, mu_h * mu_h, mu_s * mu_h
    sig_s = filt(sr * sr) - mu_s2
    sig_h = filt(hr * hr) - mu_h2
    sig_sh = filt(sr * hr) - mu_sh
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_sh + c1) * (2 * sig_sh + c2)) / (
        (mu_s2 + mu_h2 + c1) * (sig_s + sig_h + c2) + 1e-12
    )
    return float(ssim_map.mean().item())


@torch.no_grad()
def tiled_forward(model, lr: torch.Tensor, scale: int, tile: int = 128, pad: int = 16):
    """
    lr: (3, H, W) in [0,1]
    Overlapping tiles with pad; blend by hann-ish weights.
    """
    _, H, W = lr.shape
    device = next(model.parameters()).device
    # pad so dims divisible by window*something later handled by model
    out_h, out_w = H * scale, W * scale
    sr = torch.zeros(3, out_h, out_w, device=device)
    acc = torch.zeros(1, out_h, out_w, device=device)

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
            out = out[0].clamp(0, 1)
            # valid region inside patch
            ry0 = (y - y0) * scale
            rx0 = (x - x0) * scale
            ry1 = ry0 + min(tile, H - y) * scale
            rx1 = rx0 + min(tile, W - x) * scale
            # source slice in out
            sr[:, ry0 + y0 * scale : ry1 + y0 * scale, rx0 + x0 * scale : rx1 + x0 * scale] += out[
                :, ry0:ry1, rx0:rx1
            ]
            acc[:, ry0 + y0 * scale : ry1 + y0 * scale, rx0 + x0 * scale : rx1 + x0 * scale] += 1.0

    sr = sr / acc.clamp(min=1.0)
    return sr.cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=r"G:\RealSR\experiments\mod_swinir_x2\ckpt_last.pt")
    p.add_argument("--data-root", type=str, default=r"G:\RealSR\data\RealSR(V3)")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--cameras", type=str, default="Canon,Nikon")
    p.add_argument("--max-pairs", type=int, default=6)
    p.add_argument("--tile", type=int, default=128)
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
        raise SystemExit("swinir_baseline removed; use mod checkpoints only")
    else:
        model = build_model(upscale=upscale, size=model_size).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"ignore extra ckpt keys ({len(unexpected)}): {unexpected[:6]}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {ckpt_path} step={ckpt.get('step')} {arch}/{model_size} ({n_params:.2f}M) x{upscale}")

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pairs = build_realsr_index(args.data_root, cameras, "Test", (args.scale,))
    # spread across scales/cameras
    step = max(1, len(pairs) // max(args.max_pairs, 1))
    pairs = pairs[::step][: args.max_pairs]
    print(f"eval pairs: {len(pairs)}")

    out_dir = Path(args.out) if args.out else ckpt_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        lr_np = load_png(lr_path)
        hr_np = load_png(hr_path)
        # ensure hr size matches
        h, w = lr_np.shape[:2]
        hr_np = hr_np[: h * sc, : w * sc]
        lr_t = to_tensor(lr_np)
        hr_t = to_tensor(hr_np)
        sr_t = tiled_forward(model, lr_t, sc, tile=args.tile)
        # align sizes
        hh = min(sr_t.shape[1], hr_t.shape[1])
        ww = min(sr_t.shape[2], hr_t.shape[2])
        sr_c, hr_c = sr_t[:, :hh, :ww], hr_t[:, :hh, :ww]
        p_val = psnr_torch(sr_c, hr_c)
        s_val = ssim_torch(sr_c, hr_c)
        sy, hy = rgb_to_y_torch(sr_c), rgb_to_y_torch(hr_c)
        py = psnr_torch(sy, hy)
        syy = ssim_torch(sy, hy)
        name = Path(lr_path).name
        rows.append(
            {
                "name": name,
                "scale": sc,
                "psnr": p_val,
                "ssim": s_val,
                "psnr_y": py,
                "ssim_y": syy,
                "lr": lr_path,
            }
        )
        print(
            f"[{i+1}/{len(pairs)}] {name} x{sc}  RGB {p_val:.2f}/{s_val:.4f}  Y {py:.2f}/{syy:.4f}",
            flush=True,
        )
        if args.save_images:
            canvas = torch.cat(
                [
                    F.interpolate(lr_t.unsqueeze(0), scale_factor=sc, mode="bicubic", align_corners=False)[0].clamp(0,1),
                    sr_c,
                    hr_c,
                ],
                dim=2,
            )
            arr = (canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f"{Path(name).stem}_x{sc}_cmp.png")

    mean_p = float(np.mean([r["psnr"] for r in rows])) if rows else 0.0
    mean_s = float(np.mean([r["ssim"] for r in rows])) if rows else 0.0
    mean_py = float(np.mean([r["psnr_y"] for r in rows])) if rows else 0.0
    mean_sy = float(np.mean([r["ssim_y"] for r in rows])) if rows else 0.0
    summary = {
        "ckpt": str(ckpt_path),
        "scale": args.scale,
        "n": len(rows),
        "psnr": mean_p,
        "ssim": mean_s,
        "psnr_y": mean_py,
        "ssim_y": mean_sy,
        "per_image": rows,
    }
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n== MEAN over {len(rows)}: RGB {mean_p:.2f}/{mean_s:.4f}  "
        f"Y {mean_py:.2f}/{mean_sy:.4f} =="
    )
    print(f"wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
