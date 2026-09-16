"""Self-contained official RealSR Y metrics (avoid importing model package)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def rgb_to_y_matlab(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        x = rgb.astype(np.float32) / 255.0
    else:
        x = rgb.astype(np.float32)
    return 16.0 + x[..., 0] * 65.481 + x[..., 1] * 128.553 + x[..., 2] * 24.966


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    diff = a32 - b32
    mse = float(np.mean(diff * diff, dtype=np.float64))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def _gauss_win(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return np.outer(g, g)


def ssim_uint8(a: np.ndarray, b: np.ndarray, crop: int = 5) -> float:
    if crop > 0:
        a = a[crop:-crop, crop:-crop]
        b = b[crop:-crop, crop:-crop]
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    win = torch.from_numpy(_gauss_win(11, 1.5)).float().view(1, 1, 11, 11)
    ta = torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).unsqueeze(0)
    tb = torch.from_numpy(np.ascontiguousarray(b)).float().unsqueeze(0).unsqueeze(0)
    pad = 5

    def conv(x):
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), win)

    mu_a, mu_b = conv(ta), conv(tb)
    sa = conv(ta * ta) - mu_a * mu_a
    sb = conv(tb * tb) - mu_b * mu_b
    sab = conv(ta * tb) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sa + sb + c2) + 1e-12
    )
    return float(ssim_map.mean().item())


def official_pair_metrics(sr_rgb: np.ndarray, hr_rgb: np.ndarray, with_rgb_ssim: bool = False) -> dict:
    y_s = to_uint8(rgb_to_y_matlab(sr_rgb))
    y_h = to_uint8(rgb_to_y_matlab(hr_rgb))
    return {
        "psnr_y": psnr_uint8(y_s, y_h),
        "ssim_y": ssim_uint8(y_s, y_h, crop=5),
        "psnr_rgb": psnr_uint8(sr_rgb, hr_rgb),
        "ssim_rgb": (
            ssim_uint8(sr_rgb.mean(axis=2), hr_rgb.mean(axis=2), crop=5)
            if with_rgb_ssim
            else float("nan")
        ),
    }
