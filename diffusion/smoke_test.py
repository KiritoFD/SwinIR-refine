"""Local/server smoke: DiT forward + flow loss + reg objective (no VAE download).

Run before every server job:
  python -m diffusion.smoke_test
"""
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

    # --- latent flow arm: 32x32 latent, 16ch + 16ch cond = 32ch in, patch2 -> 256 tokens
    m = build_dit("S", input_size=32, patch_size=2, in_channels=32, use_checkpoint=True).to(device)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"LatentDiT-S(flow) params {n:.2f}M")
    x0 = torch.randn(2, 16, 32, 32, device=device)
    cond = torch.randn(2, 16, 32, 32, device=device)
    loss, _ = flow_loss(m, x0, cond)
    loss.backward()
    print(f"latent flow loss OK {float(loss):.4f}")
    m.eval()
    with torch.no_grad():
        s1 = sample_flow(m, (1, 16, 32, 32), cond=cond[:1], steps=4, solver="euler", device=device, seed=0)
        s2 = sample_flow(m, (1, 16, 32, 32), cond=cond[:1], steps=4, solver="euler", device=device, seed=0)
    assert torch.allclose(s1, s2), "seed does not make sampling deterministic"
    print("latent sample OK", tuple(s1.shape), "deterministic OK")
    del m, x0, cond, s1, s2
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- latent reg arm: 16ch in, zero-init head must start as identity
    mr = build_dit("S", input_size=32, patch_size=2, in_channels=16).to(device)
    zc = torch.randn(2, 16, 32, 32, device=device)
    with torch.no_grad():
        z0 = zc + mr(zc, torch.zeros(2, device=device))
    assert torch.allclose(z0, zc, atol=1e-6), "reg head is not zero-initialised"
    print("latent reg identity-init OK")
    del mr, zc, z0
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- pixel flow arm: HR 128, 6ch, patch2 -> 4096 tokens
    mp = build_dit("S", input_size=128, patch_size=2, in_channels=6, use_checkpoint=True).to(device)
    np_ = sum(p.numel() for p in mp.parameters()) / 1e6
    print(f"PixelDiT-S(flow) params {np_:.2f}M")
    x0 = torch.randn(1, 3, 128, 128, device=device)
    cond = torch.randn(1, 3, 128, 128, device=device)
    loss, _ = flow_loss(mp, x0, cond)
    loss.backward()
    mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device == "cuda" else 0
    print(f"pixel flow loss OK {float(loss):.4f} peak {mem:.2f}GB")
    del mp, x0, cond
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- pixel reg arm
    mpr = build_dit("S", input_size=128, patch_size=2, in_channels=3).to(device)
    c = torch.rand(1, 3, 128, 128, device=device)
    with torch.no_grad():
        out = (c + mpr(c, torch.zeros(1, device=device))).clamp(0, 1)
    assert torch.allclose(out, c, atol=1e-6), "pixel reg head is not zero-initialised"
    print("pixel reg identity-init OK")

    print("SMOKE PASS")


if __name__ == "__main__":
    main()
