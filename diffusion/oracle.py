"""Oracle upper-bound probes for RealSR -- measure the finite, physics-imposed
PSNR/SSIM ceiling of the dataset WITHOUT training anything:

  E_align : PSNR(GT, sub-pixel-shifted GT)   -- irreducible bi-lens misalignment.
  E_null  : PSNR(GT, ideal-low-pass(GT))      -- info above the x-scale Nyquist is
            destroyed by downsampling; no model can rebuild it.
  E_noise : PSNR(GT, denoise(GT))             -- a perfect SR matches the clean
            scene but never the GT's own sensor noise (aleatoric floor).  Uses a
            classical strong denoiser (NLM) as a proxy; a deep denoiser would read
            a slightly lower (tighter) ceiling.

All on the official Test 100 pairs, limited-range BT.601 Y, uint8, modcrop, no
shave -- the exact `official_pair_metrics` protocol.  Run on the server:

  python -m diffusion.oracle --scale 2
"""
from __future__ import annotations
import argparse, math
import numpy as np
from .data import build_index
from .metrics import official_pair_metrics
from PIL import Image


def load_hr(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def modcrop(a, m):
    h, w = a.shape[0] // m * m, a.shape[1] // m * m
    return a[:h, :w]


def to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def shift_rgb(rgb, dy, dx):
    # apply the 2-D shift per channel
    out = np.empty_like(rgb, dtype=np.float64)
    for c in range(rgb.shape[2]):
        ch = rgb[:, :, c].astype(np.float64)
        f = np.fft.fft2(ch)
        h, w = ch.shape
        ky = np.fft.fftfreq(h)[:, None]
        kx = np.fft.fftfreq(w)[None, :]
        ph = np.exp(-2j * math.pi * (ky * dy + kx * dx))
        out[:, :, c] = np.real(np.fft.ifft2(f * ph))
    return out


def ideal_lowpass(rgb, cutoff_frac):
    """Keep the central cutoff_frac of the Nyquist band (per axis), zero the rest.
    cutoff_frac in (0,1]: 1.0 = full band (identity), 0.5 = x2 downsample Nyquist."""
    out = np.empty_like(rgb, dtype=np.float64)
    h, w = rgb.shape[0], rgb.shape[1]
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt((ky / 0.5) ** 2 + (kx / 0.5) ** 2)  # normalised radius, 1.0 = Nyquist
    mask = (r <= cutoff_frac).astype(np.float64)
    for c in range(rgb.shape[2]):
        f = np.fft.fft2(rgb[:, :, c].astype(np.float64))
        out[:, :, c] = np.real(np.fft.ifft2(f * mask))
    return out


def nlms_denoise(rgb_u8, h=5.0):
    import cv2
    return cv2.fastNlMeansDenoisingColored(rgb_u8, None, h, h, 7, 21)


def mad_noise_sigma(rgb_u8):
    """Training-free, deterministic Gaussian-noise estimate sigma (in 0-255 code
    units) via Donoho's MAD on the finest orthonormal diagonal Haar subband.
    A wavelet is variance-preserving, and image structure is sparse in that band,
    so median(|HH|)/0.6745 recovers the sensor-noise std.  Averaged over RGB."""
    g = rgb_u8.astype(np.float64)
    sig = []
    for c in range(3):
        x = g[:, :, c]
        h, w = x.shape
        h -= h % 2
        w -= w % 2
        a = x[0:h:2, 0:w:2]; b = x[0:h:2, 1:w:2]
        cc = x[1:h:2, 0:w:2]; d = x[1:h:2, 1:w:2]
        hh = (a - b - cc + d) / 2.0
        sig.append(np.median(np.abs(hh)) / 0.6745)
    return float(np.mean(sig))


def _to_db(mse, peak=255.0):
    return 10.0 * math.log10(peak * peak / max(mse, 1e-9))


def mean_over(pairs, fn):
    ss, yy, n = [], [], 0
    for lr_p, hr_p, sc in pairs:
        gt = modcrop(load_hr(hr_p), 4)
        sr = modcrop(fn(gt), 4)
        h = min(gt.shape[0], sr.shape[0]); w = min(gt.shape[1], sr.shape[1])
        gt, sr = gt[:h, :w], sr[:h, :w]
        m = official_pair_metrics(sr, gt)
        ss.append(m["ssim_y"]); yy.append(m["psnr_y"]); n += 1
    return float(np.mean(ss)), float(np.mean(yy)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/RealSR(V3)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--nlm", action="store_true", help="also run the slow NLM cross-check")
    a = ap.parse_args()
    pairs = build_index(a.data_root, ("Canon", "Nikon"), "Test", a.scale)
    print(f"RealSR Test x{a.scale}: {len(pairs)} pairs; protocol=official Y/limited-range/uint8/modcrop\n")

    # E_align: sub-pixel shifts (0.2 / 0.3 / 0.5 / 1.0 px, diagonal)
    for d in (0.2, 0.3, 0.5, 1.0):
        s, y, n = mean_over(pairs, lambda g, d=d: to_u8(shift_rgb(g, d, d)))
        print(f"[E_align ] shift {d:>3} px (both axes): SSIM={s:.4f}  Y={y:.4f}  n={n}")

    # E_null: ideal low-pass at the x-scale Nyquist (representable band ceiling)
    cutoff = 1.0 / a.scale   # fraction of Nyquist the LR can carry (x2 -> 0.5)
    for cf in (cutoff, 0.75, 1.0):
        s, y, n = mean_over(pairs, lambda g, cf=cf: to_u8(ideal_lowpass(g, cf)))
        print(f"[E_null  ] low-pass cutoff {cf:.3f}*Nyquist: SSIM={s:.4f}  Y={y:.4f}  n={n}")

    # E_noise: TRAINING-FREE deterministic sensor-noise floor (Donoho MAD on the
    # diagonal Haar band) -> sigma in 0-255 code units -> PSNR ceiling.  A perfect
    # SR returns the clean scene, never the GT's own noise, so MSE >= sigma^2.
    sigs = []
    for lr_p, hr_p, _ in pairs:
        gt = modcrop(load_hr(hr_p), 4)
        sigs.append(mad_noise_sigma(gt))
    sig = float(np.mean(sigs))
    psnr_noise = _to_db(sig * sig)
    print(f"[E_noise ] MAD sensor-noise: sigma={sig:.3f} (0-255 units) -> "
          f"PSNR ceiling={psnr_noise:.3f}  n={len(pairs)}   (training-free, deterministic)")

    # Composite hard ceiling from the two MEASURED floors: band-limited structure
    # loss (E_null at x-scale) + aleatoric noise (E_noise).  MSEs add on the dB scale
    # via 10log10 sum.  Registration (E_align) is shown separately because the true
    # per-pair residual misalignment of RealSR is unknown (we only sweep sensitivity).
    _, psnr_null, _ = mean_over(pairs, lambda g: to_u8(ideal_lowpass(g, cutoff)))
    comp = -10.0 * math.log10(10 ** (-psnr_null / 10) + 10 ** (-psnr_noise / 10))
    print(f"\n[CEILING ] bandlimit+noise composite (excl. registration): Y ~ {comp:.2f} dB")
    print(f"[CEILING ] + a plausible 0.3 px residual misalignment (E_align) drags the"
          f" per-pair best-case toward the 35-37 dB band -- see E_align curve.")

    # optional cross-check with a classical (still training-free) denoiser, slow:
    if a.nlm:
        try:
            for h in (5.0, 10.0):
                s, y, n = mean_over(pairs, lambda g, h=h: nlms_denoise(g, h))
                print(f"[E_noise*] NLM h={h} cross-check: SSIM={s:.4f}  Y={y:.4f}  n={n}")
        except Exception as e:
            print(f"[E_noise*] NLM skipped: {e}")


if __name__ == "__main__":
    main()
