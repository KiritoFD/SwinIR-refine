"""Official RealSR metrics (Cai et al. ICCV 2019 Test.m).

Matches MATLAB:
  rgb2ycbcr limited-range Y, im2uint8, Cal_PSNRSSIM(..., crop=0)
"""

from __future__ import annotations

import math

import numpy as np


def modcrop(img: np.ndarray, modulo: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % modulo, : w - w % modulo]


def rgb_to_y_matlab(rgb: np.ndarray) -> np.ndarray:
    """MATLAB rgb2ycbcr Y (ITU-R BT.601 limited range).

    Input RGB float in [0,1] or uint8. Output Y in [16, 235] float64.
    """
    if rgb.dtype == np.uint8:
        x = rgb.astype(np.float64) / 255.0
    else:
        x = rgb.astype(np.float64)
    # RGB order: Y = 16 + 65.481 R + 128.553 G + 24.966 B
    return 16.0 + x[..., 0] * 65.481 + x[..., 1] * 128.553 + x[..., 2] * 24.966


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return np.outer(g, g)


def ssim_uint8(a: np.ndarray, b: np.ndarray, crop: int = 5) -> float:
    """BasicSR/MATLAB-style SSIM on single-channel uint8 (or float)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    win = _gaussian_kernel(11, 1.5)
    pad = 5

    def conv(x):
        # valid conv then crop pad like BasicSR
        H, W = x.shape
        xp = np.pad(x, pad, mode="reflect")
        out = np.zeros((H, W), dtype=np.float64)
        for i in range(11):
            for j in range(11):
                out += win[i, j] * xp[i : i + H, j : j + W]
        return out

    if crop > 0:
        a = a[crop:-crop, crop:-crop]
        b = b[crop:-crop, crop:-crop]
    mu_a, mu_b = conv(a), conv(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = conv(a * a) - mu_a2
    sb = conv(b * b) - mu_b2
    sab = conv(a * b) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2) + 1e-12)
    return float(ssim_map.mean())


def official_pair_metrics(sr_rgb: np.ndarray, hr_rgb: np.ndarray) -> dict:
    """Official protocol metrics for one pair.

    sr_rgb/hr_rgb: uint8 HWC RGB, same size (already modcropped).
    Returns official Y-PSNR/SSIM (uint8, limited-range Y) + auxiliary RGB PSNR.
    """
    y_s = to_uint8(rgb_to_y_matlab(sr_rgb))
    y_h = to_uint8(rgb_to_y_matlab(hr_rgb))
    psnr_y = psnr_uint8(y_s, y_h)
    ssim_y = ssim_uint8(y_s, y_h, crop=5)
    psnr_rgb = psnr_uint8(sr_rgb, hr_rgb)
    ssim_rgb = ssim_uint8(sr_rgb.mean(axis=2), hr_rgb.mean(axis=2), crop=5)
    return {
        "psnr_y": psnr_y,
        "ssim_y": ssim_y,
        "psnr_rgb": psnr_rgb,
        "ssim_rgb": ssim_rgb,
    }
