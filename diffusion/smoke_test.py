"""Local/server smoke: DiT forward + flow loss (no VAE download)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.dit import build_dit
from diffusion.flow import flow_loss, sample_flow


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    # latent DiT-S: 32x32 latent, 8ch (4+4 cond), patch 2 → 256 tokens
    m = build_dit("S", input_size=32, patch_size=2, in_channels=8, use_checkpoint=True).to(device)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"LatentDiT-S params {n:.2f}M")
    x0 = torch.randn(2, 4, 32, 32, device=device)
    cond = torch.randn(2, 4, 32, 32, device=device)
    loss, _ = flow_loss(m, x0, cond)
    loss.backward()
    print(f"latent flow loss OK {float(loss):.4f}")
    m.eval()
    with torch.no_grad():
        s = sample_flow(m, (1, 4, 32, 32), cond=cond[:1], steps=4, solver="euler", device=device)
    print("latent sample OK", tuple(s.shape))

    # pixel DiT-S: HR 128, 6ch, patch 2 → 4096 tokens
    mp = build_dit("S", input_size=128, patch_size=2, in_channels=6, use_checkpoint=True).to(device)
    np_ = sum(p.numel() for p in mp.parameters()) / 1e6
    print(f"PixelDiT-S params {np_:.2f}M")
    x0 = torch.randn(1, 3, 128, 128, device=device)
    cond = torch.randn(1, 3, 128, 128, device=device)
    loss, _ = flow_loss(mp, x0, cond)
    loss.backward()
    mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device == "cuda" else 0
    print(f"pixel flow loss OK {float(loss):.4f} peak {mem:.2f}GB")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
