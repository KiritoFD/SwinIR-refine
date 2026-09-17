"""Did the latent regression arm learn anything at all?

The official floors on the 100 Test pairs are:
    bicubic upsample            Y 31.7335
    VAE round-trip of bicubic   Y 31.5971   <- what latent_reg scores at step 0,
                                              because its output head is zero-init
    latent_reg_flux (trained)   Y 31.1234   <- BELOW its own initialisation

So the trained model is worse than doing nothing.  Two very different stories
fit that number:

  (a) the model learned nothing useful and its output is pure noise, or
  (b) the model learned a *correct but over-scaled* residual -- the latent L1
      objective is not the pixel PSNR objective, so the L1-optimal point can
      overshoot and land below the identity.

They are told apart by sweeping a scale on the model output:

    z = zc + alpha * model(zc, t),   alpha in {0, .25, .5, .75, 1, 1.25}

alpha = 0 is exactly the VAE round-trip.  If some alpha in (0,1) beats both 0
and 1, story (b) holds and the fix is a pixel-space loss (or a tuned scale),
not more training.  If the curve is monotonically decreasing from alpha = 0,
story (a) holds.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, "/home/ds/realsr")

import numpy as np
import torch
import torch.nn as nn

from diffusion.data import build_index, default_root
from diffusion.dit import build_dit
from diffusion.metrics import official_pair_metrics
from diffusion.vae import load_vae
from diffusion.eval_official import sr_latent_tiled, load_rgb_u8

CKPT = "/home/ds/realsr/experiments/diffusion/latent_reg_flux/ckpt_best.pt"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
MAX_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 100


class Scaled(nn.Module):
    def __init__(self, m, alpha):
        super().__init__()
        self.m = m
        self.alpha = alpha

    def forward(self, x, t):
        return self.alpha * self.m(x, t)


def main():
    scale = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    targs = ck.get("args", {}) or {}
    vae, vinfo = load_vae(ck.get("vae") or targs.get("vae") or "flux1-vae", device=device)
    in_ch = int(ck.get("in_channels") or targs.get("in_channels") or 16)
    size = targs.get("size", "S")
    patch = int(targs.get("patch", 2))
    input_size = int(ck.get("latent_size") or (targs.get("lr_patch", 256) * scale // vinfo.downscale))
    model = build_dit(size, input_size=input_size, patch_size=patch, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"ckpt {CKPT} size={size} in_ch={in_ch} input={input_size} "
          f"best_step={ck.get('step')} best_val={ck.get('best_val')}", flush=True)

    pairs = build_index(default_root(), ("Canon", "Nikon"), "Test", scale)
    if MAX_PAIRS and MAX_PAIRS < len(pairs):
        step = max(1, len(pairs) // MAX_PAIRS)
        pairs = pairs[::step][:MAX_PAIRS]
    print(f"pairs: {len(pairs)}", flush=True)

    results = {}
    for alpha in ALPHAS:
        net = Scaled(model, alpha)
        ys, ss = [], []
        for lr_path, hr_path, sc in pairs:
            lr_u8 = load_rgb_u8(lr_path)
            hr_u8 = load_rgb_u8(hr_path)
            sr = sr_latent_tiled(net, vae, vinfo, lr_u8, sc, 20, 256, 16, device,
                                 "reg", 1234, tile_batch=8)
            hh = min(sr.shape[0], hr_u8.shape[0]); ww = min(sr.shape[1], hr_u8.shape[1])
            m = official_pair_metrics(sr[:hh, :ww], hr_u8[:hh, :ww])
            ys.append(m["psnr_y"]); ss.append(m["ssim_y"])
        results[alpha] = (float(np.mean(ys)), float(np.mean(ss)))
        print(f"  alpha={alpha:<5} Y {results[alpha][0]:.4f} / SSIM {results[alpha][1]:.4f}", flush=True)

    print("\n== latent_reg_flux output-scale sweep ==")
    for a, (y, s) in results.items():
        print(f"  alpha={a:<5} Y {y:.4f} / SSIM {s:.4f}")
    Path("/home/ds/realsr/experiments/diffusion/latent_reg_alpha.json").write_text(
        json.dumps({str(k): v for k, v in results.items()}, indent=2))
    print("done")


if __name__ == "__main__":
    main()
