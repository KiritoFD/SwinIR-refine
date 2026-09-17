"""Two reference floors for the official RealSR Test protocol (100 pairs).

Why this matters
----------------
Both `reg` arms use a zero-initialised output head, so at step 0 they emit
exactly the bicubic upsample.  `pixel_reg` then climbs 31.5 -> 33.45 on its val
split (it learns ~2 dB), but `latent_reg` *falls* from its step-1000 value and
never comes back.  That makes the bicubic floor the number that decides whether
the latent regression arm learned anything at all.

  floor A  bicubic upsample, in RGB           -- the classic SR baseline
  floor B  bicubic upsample -> VAE encode -> decode (tiled, same code path as
           eval_official)                      -- what a latent reg model that
                                                  learned *nothing* would score

Run on CPU/GPU; no model checkpoint needed.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, "/home/ds/realsr")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusion.data import build_index, default_root
from diffusion.metrics import official_pair_metrics
from diffusion.vae import load_vae
from diffusion.eval_official import sr_latent_tiled, load_rgb_u8


class ZeroModel(nn.Module):
    """Identity in residual terms: z = zc + 0 -> pure VAE round-trip."""

    def forward(self, x, t):
        return torch.zeros_like(x)


def bicubic_up(lr_u8, scale):
    h, w = lr_u8.shape[:2]
    t = torch.from_numpy(np.ascontiguousarray(lr_u8)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    up = F.interpolate(t, size=(h * scale, w * scale), mode="bicubic", align_corners=False)
    return (up[0].clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).numpy()


def main():
    scale = 2
    root = default_root()
    pairs = build_index(root, ("Canon", "Nikon"), "Test", scale)
    print(f"pairs: {len(pairs)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, vinfo = load_vae("flux1-vae", device=device)
    zero = ZeroModel().to(device).eval()
    lr_patch = 256          # matches the latent reg arms' lr_patch
    input_size = lr_patch * scale // vinfo.downscale

    rows_a, rows_b = [], []
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        lr_u8 = load_rgb_u8(lr_path)
        hr_u8 = load_rgb_u8(hr_path)

        a = bicubic_up(lr_u8, sc)
        hh = min(a.shape[0], hr_u8.shape[0]); ww = min(a.shape[1], hr_u8.shape[1])
        rows_a.append(official_pair_metrics(a[:hh, :ww], hr_u8[:hh, :ww]))

        b = sr_latent_tiled(zero, vae, vinfo, lr_u8, sc, 1, lr_patch, 16, device,
                            "reg", 1234, tile_batch=8)
        hh = min(b.shape[0], hr_u8.shape[0]); ww = min(b.shape[1], hr_u8.shape[1])
        rows_b.append(official_pair_metrics(b[:hh, :ww], hr_u8[:hh, :ww]))

        if (i + 1) % 20 == 0 or i == 0:
            ma = np.mean([r["psnr_y"] for r in rows_a])
            mb = np.mean([r["psnr_y"] for r in rows_b])
            print(f"  [{i+1}/{len(pairs)}] bicubic {ma:.3f} | vae-roundtrip {mb:.3f}", flush=True)

    def mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    out = {
        "n": len(pairs),
        "bicubic": {"psnr_y": mean(rows_a, "psnr_y"), "ssim_y": mean(rows_a, "ssim_y"),
                    "psnr_rgb": mean(rows_a, "psnr_rgb")},
        "vae_roundtrip_bicubic": {"psnr_y": mean(rows_b, "psnr_y"), "ssim_y": mean(rows_b, "ssim_y"),
                                  "psnr_rgb": mean(rows_b, "psnr_rgb")},
    }
    print("\n== FLOORS (100 Test pairs, official metric) ==")
    print(f"  bicubic            Y {out['bicubic']['psnr_y']:.4f} / SSIM {out['bicubic']['ssim_y']:.4f}"
          f" / RGB {out['bicubic']['psnr_rgb']:.2f}")
    print(f"  vae round-trip     Y {out['vae_roundtrip_bicubic']['psnr_y']:.4f}"
          f" / SSIM {out['vae_roundtrip_bicubic']['ssim_y']:.4f}"
          f" / RGB {out['vae_roundtrip_bicubic']['psnr_rgb']:.2f}")
    p = Path("/home/ds/realsr/experiments/diffusion/floors.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
