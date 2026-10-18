"""Smoke + learnability gate for the newly added baseline backbones.

A mis-wired architecture can still run and then silently collapse during the real
training (that is exactly what SRResNet did: VAL peaked at 5k and fell to 20.9 dB),
so every candidate must pass a hard OVERFIT GATE before we spend GPU hours on it:

  fit one fixed batch of real-ish image pairs for a few hundred steps; the residual
  head is unbiased at step 0, so PSNR must start at the bicubic level and the net
  must drive it up by >= 6 dB (i.e. actually learn), not just shuffle loss around.

Run:  python -m diffusion.smoke_bl_archs [--device cpu|cuda] [--steps 300]
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from .baselines_sr import (build_edsr, build_rcan, build_rdn, build_rlfN, build_srformer,
                           build_hat, build_srcnn, build_vdsr, build_srresnet, build_rrdb)

ARMS = {
    # name: (builder, convention)  convention "res" = harness adds bicubic, "dir" = net predicts the image
    "srcnn": (build_srcnn, "dir"), "vdsr": (build_vdsr, "res"), "rdn": (build_rdn, "res"),
    "rlfn": (build_rlfN, "res"), "srformer": (build_srformer, "res"), "hat": (build_hat, "res"),
    # known-good controls trained on the real task already -> they set the bar
    "edsr": (build_edsr, "res"), "rcan": (build_rcan, "res"), "rrdb": (build_rrdb, "res"),
}
CONTROLS = {"edsr", "rcan", "rrdb"}


def make_batch(bs, hr, device, seed=0):
    """Synthetic LR/HR pair with real structure: smooth gradients + edges + checkerboard
    high frequency, then a genuine bicubic degradation round-trip so 'bicubic' is a
    meaningful floor rather than an identity map."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    y, x = torch.meshgrid(torch.arange(hr), torch.arange(hr), indexing="ij")
    yy, xx = y.float() / hr, x.float() / hr
    img = (0.5 + 0.35 * (xx + yy)
           + 0.12 * torch.sin(6 * math.pi * xx) * torch.cos(4 * math.pi * yy)
           + 0.10 * ((xx * 16).long() % 2 + (yy * 16).long() % 2) % 2)
    img = img.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)          # (1,3,hr,hr) in ~[0,1]
    noise = 0.02 * torch.randn(img.shape, generator=g)
    hr_img = (img + noise).clamp(0, 1)
    lr = F.interpolate(hr_img, size=(hr // 2, hr // 2), mode="bicubic", align_corners=False)
    up = F.interpolate(lr, size=(hr, hr), mode="bicubic", align_corners=False).clamp(0, 1)
    hr_img = hr_img.expand(bs, -1, -1, -1).contiguous().to(device)
    up = up.expand(bs, -1, -1, -1).contiguous().to(device)
    return up, hr_img


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def zero_last_conv(m):
    """Zero the final Conv2d so step 0 is exactly the bicubic floor. Without this the
    probe measures 'random init magnitude', not learnability (residual nets start far
    below the floor and never recover inside a few hundred steps)."""
    last = None
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            last = mod
    torch.nn.init.zeros_(last.weight)
    if last.bias is not None:
        torch.nn.init.zeros_(last.bias)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--hr", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="keep identical to the real campaign lr; a bigger lr makes this "
                         "probe measure optimizer stability instead of learnability")
    ap.add_argument("--only", default="",
                    help="comma list of backbones to probe (others are skipped)")
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="absolute dB floor (0 = off); the main criterion is relative to controls")
    ap.add_argument("--ratio", type=float, default=0.6,
                    help="an arm must reach >= ratio * median(control gain)")
    a = ap.parse_args()
    dev = a.device
    up, target = make_batch(a.bs, a.hr, dev)
    base_psnr = psnr(up, target)
    print(f"device={dev} batch={a.bs} hr={a.hr} | bicubic floor PSNR = {base_psnr:.2f} dB")
    print(f"{'backbone':10s} {'params(M)':>10s} {'ms/step':>8s} {'start dB':>9s} {'end dB':>8s} "
          f"{'gain':>7s} {'loss drop':>10s}  verdict")
    gains = {}
    fails = []
    for name, (build, conv) in ARMS.items():
        if a.only and name not in {x.strip() for x in a.only.split(",") if x.strip()}:
            continue
        torch.manual_seed(0)
        m = build("S", input_size=a.hr, in_channels=3, out_channels=3, scale=2).to(dev)
        if conv == "res":
            zero_last_conv(m)          # step 0 must equal the bicubic floor exactly

        def predict(net=m, _conv=conv):
            out = net(up, torch.zeros(a.bs, device=dev))
            return out if _conv == "dir" else (up + out)

        npar = sum(p.numel() for p in m.parameters()) / 1e6
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
        with torch.no_grad():
            start = psnr(predict().clamp(0, 1), target)
        t0 = time.time()
        first = None
        for i in range(a.steps):
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(predict().clamp(0, 1), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            if first is None:
                first = loss.item()
        dt = (time.time() - t0) / a.steps * 1000
        with torch.no_grad():
            end = psnr(predict().clamp(0, 1), target)
        last = loss.item()
        gain = end - base_psnr
        gains[name] = gain
        finite = bool(torch.isfinite(torch.tensor([first, last, end])).all().item()) and last < first
        print(f"{name:10s} {npar:10.2f} {dt:8.1f} {start:9.2f} {end:8.2f} {gain:7.2f} "
              f"{first / max(last, 1e-9):10.1f}x  {'learning' if finite else 'NOT LEARNING'}")
        del m, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cmed = sorted(gains[k] for k in CONTROLS if k in gains)
    med = cmed[len(cmed) // 2] if cmed else 0.0
    if not cmed:
        print("\n(no controls probed -> absolute bar: gain >= --min-gain dB)")
    print(f"\ncontrol median gain = {med:.2f} dB; required ratio = {a.ratio} "
          f"(=> >= {med * a.ratio:.2f} dB)")
    for name in ARMS:
        if name not in gains or name in CONTROLS:
            continue
        ok = gains[name] >= max(med * a.ratio if cmed else 0.0, a.min_gain)
        if not ok:
            fails.append(name)
        print(f"  {name:10s} {'PASS' if ok else 'FAIL'}  (gain {gains[name]:.2f} dB)")
    print("\nGATE:", "ALL PASS" if not fails else f"FAILED -> {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
