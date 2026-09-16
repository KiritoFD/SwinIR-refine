"""Estimate DiT token counts and rough VRAM for pixel vs latent diffusion on RealSR crops."""
from __future__ import annotations


def tokens(h: int, w: int, patch: int) -> int:
    return (h // patch) * (w // patch)


def attn_scores_gb(t: int, batch: int = 1, heads: int = 6, nbytes: int = 2) -> float:
    # one layer QK^T, no flash
    return batch * heads * t * t * nbytes / 1024**3


def act_gb(t: int, dim: int = 384, depth: int = 12, batch: int = 1, nbytes: int = 2, ckpt: bool = False) -> float:
    per = batch * t * dim * nbytes
    n_per_block = 4 if ckpt else 8
    return per * n_per_block * depth / 1024**3


def weights_opt_gb(millions: float = 33.0) -> float:
    # AdamW fp32-ish: param + grad + 2 states ≈ 16 B/param
    return millions * 1e6 * 16 / 1024**3


def main() -> None:
    print(f"{'cfg':28s} {'T':>7s} {'W+opt':>7s} {'act_ckpt':>9s} {'act_full':>9s} {'attn!flash':>11s}")
    rows = [
        ("pixel HR 128 p2", 128, 128, 2),
        ("pixel HR 256 p2", 256, 256, 2),
        ("pixel HR 256 p4", 256, 256, 4),
        ("pixel HR 512 p2", 512, 512, 2),
        ("latent f8 HR128 p2", 16, 16, 2),
        ("latent f8 HR256 p2", 32, 32, 2),
        ("latent f4 HR256 p2", 64, 64, 2),
        ("latent f8 HR512 p2", 64, 64, 2),
    ]
    w = weights_opt_gb()
    for name, h, wd, p in rows:
        t = tokens(h, wd, p)
        print(
            f"{name:28s} {t:7d} {w:6.2f}G "
            f"{act_gb(t, ckpt=True):8.2f}G {act_gb(t, ckpt=False):8.2f}G "
            f"{attn_scores_gb(t):10.2f}G"
        )
    print("\nbatch scales ~linearly on activations/attention; weights fixed.")
    print("DiT repo reference (latent, 256x256 img, T=256): 4090 24G @ batch 192-224 ≈ 17-20G")


if __name__ == "__main__":
    main()
