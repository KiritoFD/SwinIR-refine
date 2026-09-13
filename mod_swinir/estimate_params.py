"""Scientific parameter-budget estimate for RealSR V3 x2 SR on 8GB GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mod_swinir.model import build_model  # noqa: E402


def count_params(m) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


def swinir_configs():
    """Literature-anchored SwinIR variants (params reported by original paper)."""
    return [
        # name, embed, depths, heads, upsampler, notes
        ("SwinIR-light", 60, [2] * 6, [6] * 6, "pixelshuffledirect", "~1.19M official"),
        ("SwinIR-classical", 180, [6] * 6, [6] * 6, "pixelshuffle", "~11.90M official"),
        ("SwinIR-lightX", 60, [2, 2, 2, 2], [6] * 4, "pixelshuffledirect", "fewer blocks"),
        ("SwinIR-mid", 96, [2] * 6, [6] * 6, "pixelshuffle", "capacity mid"),
        ("SwinIR-large-ish", 120, [4] * 6, [6] * 6, "pixelshuffle", "near classical"),
    ]


def build_swinir(name):
    from models.network_swinir import SwinIR

    table = {c[0]: c for c in swinir_configs()}
    _, embed, depths, heads, upsampler, _ = table[name]
    # img_size only sets internal res tables; forward pads dynamically
    return SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=depths,
        embed_dim=embed,
        num_heads=heads,
        mlp_ratio=2.0,
        upsampler=upsampler,
        resi_connection="1conv",
    )


def estimate_band():
    """
    Capacity band from literature + data regime.

    Anchors:
      - EDSR-baseline / SwinIR-light ~1.2M: classical bicubic, often sufficient
      - SwinIR-classical / RDN ~12M: +0.3–0.5dB over light on DIV2K
      - HAT / SwinIR-L ~20M: real-world & hard degradation
      - RealSR V3: only ~400 train pairs/scale/camera (812 x2 total) + real non-bicubic
        degradation + 1–3px misregistration → needs MORE capacity than bicubic SR,
        but LESS data-limited than DIV2K (risk of overfit above ~15–20M without aug).

    Working recommendation for 8GB laptop RTX:
      - practical sweet spot: 4–12M
      - below ~2M: underfit textures / edges on real blur
      - above ~15M: diminishing returns at this data size; VRAM forces batch=1
    """
    return {
        "task": "RealSR V3 x2 (real optical degradation, registered pairs)",
        "train_pairs_scale2": 406,
        "regime": "low-data real-SR",
        "recommended_band_M": (4.0, 12.0),
        "sweet_spot_M": (8.0, 12.0),
        "hard_floor_M": 2.0,
        "diminishing_M": 15.0,
        "gpu_constraint": "RTX 4070 Laptop 8GB → base(4M)@batch8 or large(10M)@batch1",
    }


def main():
    print("=" * 72)
    print("RealSR V3 x2 — parameter budget estimate")
    print("=" * 72)
    band = estimate_band()
    for k, v in band.items():
        print(f"  {k}: {v}")

    print("\n--- SwinIR official / paper configs (actual param count) ---")
    device = "cpu"
    for name in ["SwinIR-light", "SwinIR-classical", "SwinIR-mid", "SwinIR-large-ish"]:
        try:
            m = build_swinir(name)
            print(f"  {name:18s}  {count_params(m):7.2f}M")
            del m
        except Exception as e:
            print(f"  {name:18s}  build fail: {e}")

    print("\n--- ModSwinIR configs ---")
    for size in ["tiny", "base", "large"]:
        m = build_model(2, size=size)
        n = count_params(m)
        # rough VRAM probe on CUDA if available
        vram = "n/a"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m = m.cuda()
            bs = {"tiny": 8, "base": 8, "large": 1}[size]
            ps = {"tiny": 64, "base": 64, "large": 96}[size]
            try:
                x = torch.rand(bs, 3, ps, ps, device="cuda")
                y, _ = m(x)
                loss = y.float().mean()
                loss.backward()
                vram = f"{torch.cuda.max_memory_allocated()/1024**3:.2f}GB@bs{bs},LR{ps}"
            except RuntimeError as e:
                vram = f"OOM@bs{bs}" if "memory" in str(e).lower() else str(e)[:40]
            m = m.cpu()
        print(f"  ModSwinIR-{size:5s}  {n:7.2f}M  {vram}")

    print("\n--- Recommendation ---")
    print("  Target capacity for this dataset/GPU: ~8–12M (SwinIR-classical band).")
    print("  Use SwinIR-light (1.2M) as under-capacity baseline.")
    print("  Use SwinIR-mid/classical or ModSwinIR-large (10M) as full-capacity.")
    print("  ModSwinIR-base (4.1M) sits in the practical 8GB-throughput sweet spot.")


if __name__ == "__main__":
    main()
