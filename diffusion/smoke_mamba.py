"""Smoke for the MambaSR backbone: scan exactness, identity-at-init, wiring.

  python -m diffusion.smoke_mamba

The critical check is #1: the pure-PyTorch chunked scan must EXACTLY match a
sequential reference of the S6 recurrence on adversarial shapes (L not a
multiple of the chunk size, states spanning the underflow regime).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.eval_official import sr_pixel_tiled, to_u8
from diffusion.mamba_sr import MambaCore, SS2D, _naive_scan, build_mambasr, mamba_ssm_available


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("device", device, "| mamba_ssm available:", mamba_ssm_available())

    # ---- 1. chunked scan == sequential reference ----------------------------
    core = MambaCore(d_model=12, d_state=4, expand=1, chunk=32)
    b, l = 3, 97                                   # L NOT a multiple of chunk
    xh = torch.randn(b, l, 12) * 3.0               # spread magnitudes
    dt = torch.rand(b, l, 12) * MambaCore.DT_MAX   # the regime forward() produces
    A = -torch.exp(core.A_log.float()[:12])        # (12,4) negative
    Bm = torch.randn(b, l, 4)
    C = torch.randn(b, l, 4)
    with torch.no_grad():
        y_par = core._scan(xh, dt, Bm, C)
        y_seq = _naive_scan(xh, dt, A, Bm, C)
    err = (y_par - y_seq).abs().max().item()
    scale = y_seq.abs().max().item()
    assert err / max(scale, 1e-8) < 1e-4, f"scan mismatch: {err} (rel {err/scale:.2e})"
    print(f"scan exactness OK: max abs err {err:.2e} (rel {err/scale:.2e})")

    # long sequence, many segments: compare again under heavy decay (dt = DT_MAX)
    l2 = 1000
    torch.manual_seed(1)
    x2 = torch.randn(1, l2, 12) * 3.0
    dt2 = torch.full((1, l2, 12), MambaCore.DT_MAX)   # worst-case per-step decay
    B2, C2 = torch.randn(1, l2, 4), torch.randn(1, l2, 4)
    with torch.no_grad():
        e2 = (core._scan(x2, dt2, B2, C2) - _naive_scan(x2, dt2, A, B2, C2)).abs().max().item()
    s2 = 1.0
    assert e2 / s2 < 1e-3, f"long scan mismatch: {e2}"
    print(f"long-sequence scan exactness OK (L={l2}, max |diff| {e2:.2e})")

    # ---- 2. block forward/backward ------------------------------------------
    blk = SS2D(32, d_state=16, expand=2, backend="torch").to(device)
    x = torch.randn(2, 32, 8, 8, device=device)    # (B, C, H, W) feature map
    y = blk(x)
    assert y.shape == x.shape
    y.sum().backward()
    g = blk.core.in_proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "no grad through SS2D"
    for n_, p_ in blk.named_parameters():
        assert p_.grad is None or torch.isfinite(p_.grad).all(), \
            f"non-finite grad in {n_} (fp32 scan backward broken)"
    print(f"SS2D(torch) forward/backward OK on map {tuple(x.shape)} (all grads finite)")

    # ---- 3. MambaSR identity at init ----------------------------------------
    m = build_mambasr("S", input_size=64, in_channels=3, dim=32, num_groups=2,
                      num_res=2, backend="torch").to(device)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    img = torch.rand(1, 3, 64, 64, device=device)
    t0 = torch.zeros(1, device=device)
    with torch.no_grad():
        y = m(img, t0)
        trunk = m.stem(img)
        for g_ in m.groups:
            trunk = g_(trunk)
    assert y.abs().max().item() == 0.0, "output head is not zero-init"
    assert (trunk - m.stem(img)).abs().max().item() < 1e-5, \
        "trunk is not identity at init (block proj / RG conv zero-init broken)"
    print(f"MambaSR init OK: output == 0, trunk == identity; dim=32/2x2 params {n:.2f}M")

    # ---- 4. parameter table for the A/B arms --------------------------------
    for name, kw in (
        ("c128 4x4 (spec)", dict(dim=128, num_groups=4, num_res=4)),
        ("c192 4x4", dict(dim=192, num_groups=4, num_res=4)),
        ("c256 4x4", dict(dim=256, num_groups=4, num_res=4)),
    ):
        mm = build_mambasr("S", input_size=128, in_channels=3, backend="torch", **kw)
        print(f"  {name}: {sum(p.numel() for p in mm.parameters()) / 1e6:.2f}M")
        del mm

    # ---- 5. tiled eval identity ---------------------------------------------
    import numpy as np

    mt = build_mambasr("S", input_size=64, in_channels=3, dim=32, num_groups=2,
                       num_res=1, backend="torch").to(device)
    lr_u8 = (np.random.default_rng(1).random((24, 20, 3)) * 255).astype(np.uint8)
    sr = sr_pixel_tiled(mt, lr_u8, 2, steps=8, tile=16, pad=8, residual=False,
                        device=device, objective="reg", seed=0)
    lr_t = torch.from_numpy(lr_u8).permute(2, 0, 1).float()[None] / 255.0
    expect = F.interpolate(lr_t, size=(48, 40), mode="bicubic", align_corners=False).clamp(0, 1)
    diff = int((sr.astype(int) - to_u8(expect[0]).astype(int)).__abs__().max())
    assert diff == 0, f"tiled eval moved pixels by {diff}"
    print(f"tiled eval + mamba: zero-init model reproduces bicubic exactly (shape {sr.shape})")

    # ---- 6. tiny train through the real trainer -----------------------------
    root = next((r for r in (r"G:\RealSR\data\RealSR(V3)", "/home/ds/realsr/data/RealSR(V3)",
                             "data/RealSR(V3)") if Path(r).is_dir()), None)
    if root is None:
        print("RealSR data not found -- skipping trainer smoke")
        print("SMOKE PASS (module-level only)")
        return
    sys.argv = ["train_pixel", "--out", "experiments/smoke_mamba", "--backbone", "mamba",
                "--objective", "reg", "--base", "32", "--num-groups", "2", "--num-res", "2",
                "--lr-patch", "32", "--steps", "2", "--batch", "2", "--eval-every", "0",
                "--num-workers", "0", "--device", device]
    import diffusion.train_pixel as tp

    try:
        tp.main()
    finally:
        sys.argv = ["train_pixel"]
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
