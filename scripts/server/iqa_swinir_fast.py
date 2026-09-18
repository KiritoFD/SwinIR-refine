"""MUSIQ / MANIQA for the SwinIR-family baselines, done properly.

The earlier attempt reused `model/eval.py`'s tiled_forward, which accumulates the
SR image in a *CPU* tensor and copies every tile back to host:
    sr = torch.zeros(3, H, W)                 # CPU
    out = out[0].clamp(0,1).float().cpu()     # per-tile GPU->CPU
    sr[dst] += out[src]                       # CPU scatter-add
That was written around an 8GB card ("2k+ SR tensors on an 8GB card OOM").  On a
48GB card it is pure overhead, and at tile=96 a 1000x1400 input is ~150 tiles of
host round-trips per image.  This version accumulates on the GPU and uses tile
256 (~24 tiles), so a whole model takes minutes instead of ~40.

Everything else matches model/eval.py: modcrop by 4, same index, same metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/ds/realsr")

import numpy as np
import torch
from PIL import Image

from model.dataset import build_realsr_index
from model.metrics import modcrop, official_pair_metrics
from model.model import build_model

from diffusion.iqa import IQAScorer

CKPTS = {
    "E1_swinir_light": "experiments/matrix_x2/E1_swinir_light/ckpt_best.pt",
    "E2_swinir_capmatch": "experiments/matrix_x2/E2_swinir_capmatch/ckpt_best.pt",
    "A0_l1_only": "experiments/ablation_x2/A0_l1_only/ckpt_best.pt",
    "E11_align_loss": "experiments/improve/E11_align_loss/ckpt_best.pt",
}


def load_rgb_u8(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@torch.no_grad()
def tiled_forward_gpu(model, lr: torch.Tensor, scale: int, tile: int = 256, pad: int = 16):
    """lr: (1,3,H,W) on device.  Accumulate everything on the GPU, return uint8 HWC."""
    _, _, H, W = lr.shape
    device = lr.device
    out_h, out_w = H * scale, W * scale
    sr = torch.zeros(1, 3, out_h, out_w, device=device, dtype=torch.float32)
    acc = torch.zeros(1, 1, out_h, out_w, device=device, dtype=torch.float32)

    ys = list(range(0, max(H - tile, 0) + 1, tile))
    if not ys or ys[-1] + tile < H:
        ys.append(max(H - tile, 0))
    xs = list(range(0, max(W - tile, 0) + 1, tile))
    if not xs or xs[-1] + tile < W:
        xs.append(max(W - tile, 0))

    for y in ys:
        for x in xs:
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(H, y + tile + pad), min(W, x + tile + pad)
            out, _ = model(lr[:, :, y0:y1, x0:x1])
            out = out[0:1].clamp(0, 1).float()
            ry0, rx0 = (y - y0) * scale, (x - x0) * scale
            ry1, rx1 = ry0 + min(tile, H - y) * scale, rx0 + min(tile, W - x) * scale
            sr[:, :, ry0 + y0 * scale:ry1 + y0 * scale, rx0 + x0 * scale:rx1 + x0 * scale] += out[:, :, ry0:ry1, rx0:rx1]
            acc[:, :, ry0 + y0 * scale:ry1 + y0 * scale, rx0 + x0 * scale:rx1 + x0 * scale] += 1.0
    sr = sr / acc.clamp(min=1.0)
    return (sr.clamp(0, 1) * 255.0).round().byte()[0].permute(1, 2, 0).cpu().numpy()


p = argparse.ArgumentParser()
p.add_argument("--data-root", default="data/RealSR(V3)")
p.add_argument("--scale", type=int, default=2)
p.add_argument("--tile", type=int, default=256)
p.add_argument("--device", default="cuda")
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()

device = torch.device(args.device)
scorer = IQAScorer(device)
scorer.available()
pairs = build_realsr_index(args.data_root, ("Canon", "Nikon"), "Test", (args.scale,))
if args.limit:
    pairs = pairs[: args.limit]
print(f"{len(pairs)} pairs", flush=True)

for label, rel in CKPTS.items():
    ckpt_path = Path("/home/ds/realsr") / rel
    out_dir = Path("/home/ds/realsr") / "experiments" / "diffusion" / "iqa_eval" / label
    if (out_dir / "eval.json").exists():
        print(f"{label}: already done", flush=True)
        continue
    if not ckpt_path.is_file():
        print(f"{label}: NO CKPT", flush=True)
        continue

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    meta = ckpt.get("meta", {})
    upscale = targs.get("scale", args.scale)
    size = targs.get("model_size", "base")
    arch = targs.get("arch", meta.get("arch", "mod"))
    if arch == "swinir":
        from model.swinir_baseline import SwinIRTrainWrapper, build_swinir

        model = SwinIRTrainWrapper(build_swinir(size=size, upscale=upscale)).to(device)
    else:
        model = build_model(upscale=upscale, size=size).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    n_params = sum(q.numel() for q in model.parameters()) / 1e6
    print(f"\n=== {label}: {arch}/{size} {n_params:.2f}M x{upscale} ===", flush=True)

    rows = []
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        lr_u8 = modcrop(load_rgb_u8(lr_path), 4)
        hr_u8 = modcrop(load_rgb_u8(hr_path), 4)
        h, w = lr_u8.shape[:2]
        hr_u8 = hr_u8[: h * sc, : w * sc]
        lr_t = torch.from_numpy(np.ascontiguousarray(lr_u8)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        sr_u8 = tiled_forward_gpu(model, lr_t, sc, tile=args.tile)
        hh = min(sr_u8.shape[0], hr_u8.shape[0])
        ww = min(sr_u8.shape[1], hr_u8.shape[1])
        m = official_pair_metrics(sr_u8[:hh, :ww], hr_u8[:hh, :ww])
        m.update(scorer.score(sr_u8[:hh, :ww]))
        m["name"] = Path(lr_path).name
        rows.append(m)
        del lr_t
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(pairs)}] Y {m['psnr_y']:.2f}", flush=True)

    def mean(k):
        v = [r[k] for r in rows if k in r and r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")

    summary = {
        "ckpt": str(ckpt_path), "arch": arch, "model_size": size, "params_M": n_params,
        "n": len(rows),
        "psnr_y": mean("psnr_y"), "ssim_y": mean("ssim_y"), "psnr_rgb": mean("psnr_rgb"),
        "iqa_scores": {n: mean(n) for n in scorer.available()},
        "per_image": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2))
    print(f"  {label}: Y {summary['psnr_y']:.4f} SSIM {summary['ssim_y']:.4f} {summary['iqa_scores']}",
          flush=True)
    del model
    torch.cuda.empty_cache()
