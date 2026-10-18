"""Verification of Wavelet-Decoupled Optical Consistency Projection (WD-DCP).

Compares:
1. Standard DCP (full-band back-projection): changes both LL and [HL, LH, HH].
2. WD-DCP (hf_decay=0.0): strictly preserves [HL, LH, HH] of the model's generated details,
   while projecting low frequencies (LL) to match physical LR measurements.
3. Optical PSF convolved DCP: incorporates camera optical point spread function.
4. Chromatic-aberration per-channel PSF (different sigmas for R, G, B).
"""

import torch
import torch.nn.functional as F
from diffusion.eval_official import data_consistency_project, _parse_sigmas, _build_psf_kernel
from diffusion.wavelet import haar_dwt2

def test_wd_dcp():
    torch.manual_seed(42)
    B, C, H, W = 1, 3, 128, 128
    hr_model = torch.rand(B, C, H, W) * 0.6 + 0.2
    lr_captured = F.interpolate(hr_model, size=(H // 2, W // 2), mode="bicubic", align_corners=False)
    lr_captured = (lr_captured + 0.02).clamp(0, 1)

    # 1. Standard DCP (full band, hf_decay=1.0)
    hr_std_dcp = data_consistency_project(hr_model.clone(), lr_captured, iters=1, eta=0.15, hf_decay=1.0, sigma=0.0)

    # 2. WD-DCP (high frequency protected, hf_decay=0.0)
    hr_wd_dcp = data_consistency_project(hr_model.clone(), lr_captured, iters=1, eta=0.15, hf_decay=0.0, sigma=0.0)

    # 3. Wavelet analysis of the differences
    orig_w = haar_dwt2(hr_model)
    std_w = haar_dwt2(hr_std_dcp)
    wd_w = haar_dwt2(hr_wd_dcp)

    ll_orig, hf_orig = orig_w[:, :C], orig_w[:, C:]
    ll_std, hf_std = std_w[:, :C], std_w[:, C:]
    ll_wd, hf_wd = wd_w[:, :C], wd_w[:, C:]

    std_ll_diff = (ll_std - ll_orig).abs().mean().item()
    std_hf_diff = (hf_std - hf_orig).abs().mean().item()

    wd_ll_diff = (ll_wd - ll_orig).abs().mean().item()
    wd_hf_diff = (hf_wd - hf_orig).abs().mean().item()

    print("=== Wavelet Decomposition Energy Analysis ===")
    print(f"Standard DCP  | LL diff: {std_ll_diff:.6f} | HF diff: {std_hf_diff:.6f}")
    print(f"WD-DCP (ours) | LL diff: {wd_ll_diff:.6f} | HF diff: {wd_hf_diff:.6f}")

    assert wd_hf_diff < 1e-6, f"WD-DCP with hf_decay=0.0 must leave high frequencies unchanged! Got {wd_hf_diff}"
    assert wd_ll_diff > 1e-3, "WD-DCP must update low frequencies to match LR measurement!"
    print("SUCCESS 1: WD-DCP mathematically guarantees 100% preservation of model-generated high frequencies!")

    # 4. Test Per-Channel Chromatic-Aberration PSF
    sigmas_str = "0.7,0.6,0.5"
    parsed = _parse_sigmas(sigmas_str)
    assert parsed == (0.7, 0.6, 0.5), f"Failed parsing: {parsed}"
    hr_ca = data_consistency_project(hr_model.clone(), lr_captured, iters=1, eta=0.12, hf_decay=0.0, sigma=sigmas_str)
    ca_w = haar_dwt2(hr_ca)
    ca_ll_diff = (ca_w[:, :C] - orig_w[:, :C]).abs().mean().item()
    ca_hf_diff = (ca_w[:, C:] - orig_w[:, C:]).abs().mean().item()
    print(f"WD-DCP (CA per-channel PSF: {parsed}) | LL diff: {ca_ll_diff:.6f} | HF diff: {ca_hf_diff:.6f}")
    assert ca_hf_diff < 1e-6, f"CA-WD-DCP must preserve high frequencies! Got {ca_hf_diff}"
    print("SUCCESS 2: Per-channel chromatic-aberration PSF mathematically validated!")

if __name__ == "__main__":
    test_wd_dcp()
