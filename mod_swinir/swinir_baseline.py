"""SwinIR baseline adapter (official models/network_swinir.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.network_swinir import SwinIR  # noqa: E402


def build_swinir(size: str = "light", upscale: int = 2) -> nn.Module:
    """
    size:
      light     — official lightweight (~1.19M), pixelshuffledirect
      mid       — embed96 depths2x6 (~5.5M)
      classical — official classical (~11.9M), pixelshuffle
      largeish  — embed120 depths4x6 (~10M), pixelshuffle
    """
    table = {
        "light": dict(embed_dim=60, depths=[2] * 6, num_heads=[6] * 6, upsampler="pixelshuffledirect"),
        "mid": dict(embed_dim=96, depths=[2] * 6, num_heads=[6] * 6, upsampler="pixelshuffle"),
        "classical": dict(embed_dim=180, depths=[6] * 6, num_heads=[6] * 6, upsampler="pixelshuffle"),
        "largeish": dict(embed_dim=120, depths=[4] * 6, num_heads=[6] * 6, upsampler="pixelshuffle"),
    }
    cfg = table[size]
    return SwinIR(
        upscale=upscale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        mlp_ratio=2.0,
        resi_connection="1conv",
        **cfg,
    )


class SwinIRTrainWrapper(nn.Module):
    """Match ModSwinIR interface: forward(lr) -> (sr, log_var|None)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, lr):
        sr = self.net(lr)
        return sr, None
