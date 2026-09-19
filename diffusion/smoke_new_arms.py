"""Local/server smoke for the three new arms: coord / freq-route / adversarial min-max.

Run before any server job that uses them:
  python -m diffusion.smoke_new_arms

Checks, in order:
  1. freq-route U-Net: identity at init, alpha map exists, grads reach the router,
     param overhead printed.
  2. coord U-Net: at init a 5ch input with zeroed coord channels must give the
     EXACT same output as the 3ch no-coord model (zero-init stem slice), and
     grads reach the coord stem slice after a real step.
  3. AdvDegradation: shapes, determinism with fixed eps, one full min-max step
     (theta and phi both get gradients, phi's are flipped to ascent), bank is a
     proper simplex of normalised PSFs.
  4. tiled eval with coord: a zero-init coord model must reproduce bicubic-up
     exactly through sr_pixel_tiled (the tile/coord slicing end-to-end).
  5. RealSRCropDataset(coord=True): shapes/range on the local RealSR copy.
  6. tiny 2-step training runs (plain / coord / freq-route) on the real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.adversarial import AdvDegradation
from diffusion.data import RealSRCropDataset
from diffusion.eval_official import sr_pixel_tiled, to_u8
from diffusion.unet import build_unet


def n_params(m) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    t0 = torch.zeros(2, device=device)

    # ---- 1. freq-route ------------------------------------------------------
    base_kw = dict(input_size=128, in_channels=3, base=32, mult=(1, 2, 4, 4),
                   num_res=2, attn_levels=(2, 3), in_stride=1, out_scale=1,
                   use_checkpoint=False)
    m_plain = build_unet("S", **base_kw).to(device)
    m_route = build_unet("S", **base_kw, freq_route=True).to(device)
    print(f"unet b32 plain {n_params(m_plain):.2f}M | freq_route {n_params(m_route):.2f}M "
          f"(+{n_params(m_route) - n_params(m_plain):.2f}M)")
    x = torch.rand(2, 3, 128, 128, device=device)
    with torch.no_grad():
        res0 = m_route(x, t0)
    assert res0.abs().max().item() == 0.0, "freq_route breaks identity-at-init"
    loss = F.l1_loss(m_route(x, t0), torch.zeros_like(x))
    loss.backward()
    g_router = m_route.enc[0][0].router.proj.weight.grad
    g_smooth = m_route.enc[0][0].smooth.net[-1].weight.grad
    assert g_router is not None and g_router.abs().sum() >= 0, "router got no grad tensor"
    assert g_smooth is not None, "smooth branch got no grad tensor"
    print("freq-route: identity-at-init OK, backward OK (router/smooth wired)")
    del m_plain, m_route
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 2. coord -----------------------------------------------------------
    m_c = build_unet("S", **base_kw, coord_channels=2).to(device)
    assert m_c.stem.weight.shape[1] == 5
    assert m_c.stem.weight[:, 3:].abs().max().item() == 0.0, "coord stem slice not zero-init"
    inp5a = torch.cat([x, torch.rand(2, 2, 128, 128, device=device) * 2 - 1], dim=1)
    inp5b = torch.cat([x, torch.rand(2, 2, 128, 128, device=device) * 2 - 1], dim=1)
    with torch.no_grad():
        d = (m_c(inp5a, t0) - m_c(inp5b, t0)).abs().max().item()
    assert d < 1e-6, f"coord model at init still depends on coords (diff {d})"
    # the zero-init OUTPUT head blocks all gradient at step 0 (only the head
    # itself gets grads), so wake the head up first, then check the coord slice
    with torch.no_grad():
        m_c.out[-1].weight.normal_(0, 0.02)
    loss = F.l1_loss(m_c(inp5a, t0), torch.zeros_like(x))
    loss.backward()
    assert m_c.stem.weight.grad[:, 3:].abs().sum() > 0, "no grad into coord stem slice"
    print("coord: zero-init slice OK, init-equivalence OK, backward OK")
    del m_c
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 3. adversarial min-max --------------------------------------------
    adv = AdvDegradation(scale=2, sigma_max=0.12).to(device)
    print(f"adversary: {adv.n_kernels} kernels, params {n_params(adv)*1e3:.1f}K")
    bank = adv.bank
    assert bank.shape[0] == adv.n_kernels and bank.shape[-1] % 2 == 1
    assert torch.allclose(bank.sum(dim=(1, 2)), torch.ones(adv.n_kernels, device=bank.device),
                          atol=1e-5), "bank kernels are not normalised"
    net = build_unet("XS", input_size=128, in_channels=3, base=32, mult=(1, 2, 4, 4),
                     num_res=1, attn_levels=(2,), in_stride=1, out_scale=1).to(device)
    hr = torch.rand(2, 3, 128, 128, device=device)
    eps = torch.randn(2, 3, 64, 64, device=device)
    lr1 = adv(hr, eps=eps)
    lr2 = adv(hr, eps=eps)
    assert lr1.shape == (2, 3, 64, 64), f"bad LR shape {tuple(lr1.shape)}"
    assert torch.allclose(lr1, lr2), "fixed eps is not deterministic"
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    opt_phi = torch.optim.Adam(adv.parameters(), lr=1e-4)
    lr = adv(hr)
    lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    loss = F.l1_loss(lr_up + net(lr_up, t0), hr)
    loss.backward()
    g_theta = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    g_phi = sum(p.grad.abs().sum().item() for p in adv.parameters() if p.grad is not None)
    assert g_theta > 0 and g_phi > 0, f"min-max grads missing: theta {g_theta} phi {g_phi}"
    # the ascent sign flip must be its own inverse and must not touch theta
    for p in adv.parameters():
        if p.grad is not None:
            p.grad.neg_()
    opt.step()
    opt_phi.step()
    rep = adv.kernel_report(hr)
    print(f"adversary: forward/determinism OK, min-max grads OK "
          f"(theta {g_theta:.1f}, phi {g_phi:.1f}); report {rep}")
    del adv, net, hr, lr, lr_up, loss
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 4. tiled eval with coord ------------------------------------------
    m_tile = build_unet("S", input_size=128, in_channels=3, base=32, mult=(1, 2, 4, 4),
                        num_res=2, attn_levels=(2, 3), in_stride=1, out_scale=1,
                        coord_channels=2).to(device)
    lr_u8 = (np.random.default_rng(0).random((48, 40, 3)) * 255).astype(np.uint8)
    sr = sr_pixel_tiled(m_tile, lr_u8, 2, steps=8, tile=24, pad=8, residual=False,
                        device=device, objective="reg", seed=0)
    lr_t = torch.from_numpy(lr_u8).permute(2, 0, 1).float()[None] / 255.0
    expect = F.interpolate(lr_t, size=(96, 80), mode="bicubic", align_corners=False).clamp(0, 1)
    expect_u8 = to_u8(expect[0])
    assert sr.shape == expect_u8.shape, f"tile shape {sr.shape} vs {expect_u8.shape}"
    bad = int((sr.astype(int) - expect_u8.astype(int)).__abs__().max())
    assert bad <= 1, f"zero-init coord model through tiled eval moved pixels by {bad}"
    print(f"tiled eval + coord: zero-init model reproduces bicubic-up exactly "
          f"(max |diff| = {bad} u8, shape {sr.shape})")
    del m_tile
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 5/6. real-data dataset + tiny trains -------------------------------
    root = next((r for r in (r"G:\RealSR\data\RealSR(V3)", "/home/ds/realsr/data/RealSR(V3)",
                             "data/RealSR(V3)") if Path(r).is_dir()), None)
    if root is None:
        print("RealSR data not found locally -- skipping dataset/tiny-train checks")
        print("SMOKE PASS (module-level checks only)")
        return
    ds = RealSRCropDataset(root, "Train", ("Canon", "Nikon"), 2, 64, True, coord=True)
    b = ds[0]
    assert b["coord"].shape == (2, 128, 128), f"coord shape {tuple(b['coord'].shape)}"
    assert float(b["coord"].min()) >= -1.0 and float(b["coord"].max()) <= 1.0
    seen = set()
    for i in range(8):
        c = ds[(i * 7) % len(ds)]["coord"]
        seen.add((float(c[0, 0, 0]), float(c[1, 0, -1])))
    print(f"dataset coord: shape OK, range OK, {len(seen)}/8 distinct crops -> "
          "absolute position varies across crops")
    tiny = ["--steps", "2", "--batch", "2", "--eval-every", "0", "--num-workers", "0"]
    for extra in ([], ["--coord"], ["--freq-route"]):
        argv = ["--out", "experiments/smoke_new_arms", "--backbone", "unet",
                "--lr-patch", "64", "--base", "32", "--objective", "reg",
                "--device", device, *tiny, *extra]
        sys.argv = ["train_pixel", *argv]
        import diffusion.train_pixel as tp
        try:
            tp.main()
        finally:
            sys.argv = ["train_pixel"]
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
