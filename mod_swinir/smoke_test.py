"""Quick component + end-to-end smoke test on GPU."""

from __future__ import annotations

import time

import torch

from mod_swinir.dataset import RealSRPairDataset
from mod_swinir.layers import (
    AmplitudeModulatedFourierFFN,
    DeformAlign,
    GatedDconvFFN,
    OverlappingWindowAttention,
    RadialAwarePE,
)
from mod_swinir.losses import UWCLLoss, psnr
from mod_swinir.model import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("total mem GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    B, C, H, W = 2, 48, 48, 48
    x = torch.randn(B, C, H, W, device=device)

    # layers
    amf = AmplitudeModulatedFourierFFN(C).to(device)
    gdfn = GatedDconvFFN(C).to(device)
    dcn = DeformAlign(C).to(device)
    rape = RadialAwarePE(C).to(device)
    oca = OverlappingWindowAttention(C, window_size=8, overlap=2, num_heads=3).to(device)

    y = amf(x)
    assert y.shape == x.shape, y.shape
    y = gdfn(x)
    assert y.shape == x.shape, y.shape
    y = dcn(x)
    assert y.shape == x.shape, y.shape
    pe = rape(B, H, W, device, torch.float32)
    assert pe.shape == (B, H, W, C), pe.shape
    xh = x.permute(0, 2, 3, 1)
    y = oca(xh)
    assert y.shape == xh.shape, y.shape
    print("layer shapes OK")

    # model forward/backward
    model = build_model(upscale=2, size="tiny").to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params {n_params:.2f}M")

    lr = torch.rand(B, 3, 48, 48, device=device)
    hr = torch.rand(B, 3, 96, 96, device=device)
    sr, log_var = model(lr)
    print("sr", tuple(sr.shape), "log_var", tuple(log_var.shape) if log_var is not None else None)
    assert sr.shape == hr.shape
    assert log_var is not None and log_var.shape == (B, 1, 96, 96)

    crit = UWCLLoss()
    total, parts = crit(sr, log_var, hr)
    total.backward()
    print("loss", float(total), {k: float(v) for k, v in parts.items()})
    print("psnr", float(psnr(sr, hr)))
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
    print(f"peak VRAM after model step: {peak:.2f} GB")

    # dataset
    ds = RealSRPairDataset(
        r"G:\RealSR\data\RealSR(V3)",
        split="Train",
        cameras=("Canon",),
        scales=(2,),
        lr_patch=48,
        max_pairs=4,
    )
    item = ds[0]
    print("data lr", tuple(item["lr"].shape), "hr", tuple(item["hr"].shape), "scale", item["scale"])
    assert item["lr"].shape[1] == 48 and item["hr"].shape[1] == 96

    # short train-like loop
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    t0 = time.time()
    for i in range(3):
        lr_b = item["lr"].unsqueeze(0).to(device).repeat(B, 1, 1, 1)
        hr_b = item["hr"].unsqueeze(0).to(device).repeat(B, 1, 1, 1)
        sr, log_var = model(lr_b)
        total, _ = crit(sr, log_var, hr_b)
        optim.zero_grad(set_to_none=True)
        total.backward()
        optim.step()
        print(f"train iter {i} loss={float(total):.4f} psnr={float(psnr(sr, hr_b)):.2f}")
    print(f"train loop {time.time()-t0:.2f}s")
    if device.type == "cuda":
        print(f"final peak VRAM {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
