"""Unit tests for physical-informed supervisory signals in diffusion.physics_loss."""

import torch
import torch.nn.functional as F
from diffusion.physics_loss import (
    ll_swap_target,
    bounded_align_target,
    optical_consistency_loss,
    chromatic_aberration_loss,
)
from diffusion.wavelet import haar_dwt2


def test_physics_losses():
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64

    # 1. Test LL-Swap (exact mathematical bijection test)
    hr = torch.rand(B, C, H, W)
    lr_up = torch.rand(B, C, H, W)
    swapped = ll_swap_target(lr_up, hr, clamp=False)
    assert swapped.shape == (B, C, H, W)

    w_swap = haar_dwt2(swapped)
    w_lr = haar_dwt2(lr_up)
    w_hr = haar_dwt2(hr)

    # LL subband of swapped must equal LL subband of lr_up
    ll_diff = (w_swap[:, :C] - w_lr[:, :C]).abs().mean().item()
    # HF subbands of swapped must equal HF subbands of hr
    hf_diff = (w_swap[:, C:] - w_hr[:, C:]).abs().mean().item()
    print(f"LL-Swap Test (exact) | LL diff from lr_up: {ll_diff:.6f} | HF diff from hr: {hf_diff:.6f}")
    assert ll_diff < 1e-5, f"LL must match lr_up! Got {ll_diff}"
    assert hf_diff < 1e-5, f"HF must match hr! Got {hf_diff}"
    print("SUCCESS: ll_swap_target verified!")

    # 2. Test Bounded Align
    pred = hr.clone()
    aligned = bounded_align_target(pred, hr, max_shift=0.5)
    assert aligned.shape == hr.shape
    l1_raw = (pred - hr).abs().mean().item()
    l1_aligned = (pred - aligned).abs().mean().item()
    print(f"Bounded-Align Test | L1 raw: {l1_raw:.6f} | L1 aligned: {l1_aligned:.6f}")
    assert l1_aligned <= l1_raw + 1e-6
    print("SUCCESS: bounded_align_target verified!")

    # 3. Test Optical Consistency Loss
    lr = F.interpolate(hr, size=(H // 2, W // 2), mode="bicubic", align_corners=False)
    opt_loss = optical_consistency_loss(hr, lr, sigma=0.6)
    print(f"Optical Consistency Loss: {opt_loss.item():.6f}")
    assert opt_loss.item() >= 0.0
    print("SUCCESS: optical_consistency_loss verified!")

    # 4. Test Chromatic Aberration Loss
    ca_loss = chromatic_aberration_loss(hr)
    print(f"Chromatic Aberration Loss: {ca_loss.item():.6f}")
    assert ca_loss.item() >= 0.0
    print("SUCCESS: chromatic_aberration_loss verified!")

    # 5. Test YCbCr & Chroma TV Loss
    from diffusion.physics_loss import rgb_to_ycbcr_601, chroma_tv_loss, ycbcr_weighted_l1_loss
    ycbcr = rgb_to_ycbcr_601(hr)
    assert ycbcr.shape == hr.shape
    assert ycbcr[:, 0:1].min() >= 0.0 and ycbcr[:, 0:1].max() <= 1.0

    tv_loss = chroma_tv_loss(hr)
    print(f"Chroma TV Loss: {tv_loss.item():.6f}")
    assert tv_loss.item() >= 0.0

    ycbcr_loss = ycbcr_weighted_l1_loss(pred, hr, y_weight=1.0, chroma_weight=0.2)
    print(f"YCbCr Weighted Loss (self): {ycbcr_loss.item():.6f}")
    assert ycbcr_loss.item() < 1e-6
    print("SUCCESS: YCbCr and Chroma TV losses verified!")


if __name__ == "__main__":
    test_physics_losses()
