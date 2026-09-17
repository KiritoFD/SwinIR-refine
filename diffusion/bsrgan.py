"""BSRGAN-style high-order degradation (Zhang et al., ICCV 2021).

This is the degradation SwinIR uses for its real-world SR model ("We use the
pioneering practical degradation model from BSRGAN"). We need it because the
published SwinIR/BSRGAN models are trained on DIV2K+Flickr2K(+OST) with
*synthesised* LR images, not on RealSR pairs.

Re-implemented with torch + PIL only: the server has no cv2 and no scipy.

Recipe (per BSRGAN paper / reference code):
  * apply p rounds (p in {1,2,3}) of {blur, resize, noise, jpeg} in a
    RANDOMLY SHUFFLED order -- the random shuffle is the main idea, it stops
    the network from learning one fixed degradation order
  * blur is drawn from a mixed kernel pool: isotropic / anisotropic Gaussian,
    generalised (sharper) and plateau (flatter) variants, plus a sinc filter
    that injects ringing / overshoot artefacts
  * finally downsample by sf with a random interpolation

Everything is scaled so the same code works for x2 (our RealSR task) and x4
(SwinIR/BSRGAN's canonical setting).
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# ----------------------------------------------------------------- kernels


def _meshgrid2d(ksize: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    half = (ksize - 1) / 2.0
    ax = torch.arange(ksize, device=device, dtype=torch.float32) - half
    ys, xs = torch.meshgrid(ax, ax, indexing="ij")
    return xs, ys


def gaussian_kernel_2d(sigma_x: float, sigma_y: float, theta: float, ksize: int) -> torch.Tensor:
    xs, ys = _meshgrid2d(ksize, torch.device("cpu"))
    ct, st = math.cos(theta), math.sin(theta)
    u = ct * xs + st * ys
    v = -st * xs + ct * ys
    sx = max(sigma_x, 1e-3)
    sy = max(sigma_y, 1e-3)
    k = torch.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return k / k.sum()


def sinc_kernel(omega_c: float, ksize: int) -> torch.Tensor:
    """2D separable sinc -- models ring / overshoot artefacts."""
    half = (ksize - 1) / 2.0
    ax = torch.arange(ksize, dtype=torch.float32) - half
    ax = ax + 1e-5
    k1d = omega_c / math.pi * torch.sinc(omega_c * ax / math.pi)
    k = torch.outer(k1d, k1d)
    return k / k.sum()


def _random_kernel(sf: int, rng: random.Random) -> torch.Tensor:
    """Mixed kernel pool, sigma ranges scaled with the scale factor."""
    # BSRGAN's [0.1, 2.4] is quoted for x4; scale linearly with sf so that x2
    # is not hit with a relatively twice-as-strong blur.
    base = sf / 4.0
    sigma_range = (max(0.1 * base, 0.05), max(2.4 * base, 0.05))
    ksize = rng.choice([7, 9, 11, 13, 15, 17, 19, 21])

    kind = rng.random()
    if kind < 0.4:                      # isotropic
        s = rng.uniform(*sigma_range)
        k = gaussian_kernel_2d(s, s, 0.0, ksize)
    elif kind < 0.8:                    # anisotropic
        s = rng.uniform(*sigma_range)
        k = gaussian_kernel_2d(s, rng.uniform(0.15, 1.5) * s, rng.uniform(0, math.pi), ksize)
    else:                               # plateau-ish (flatter top, sharper fall)
        s = rng.uniform(*sigma_range)
        k = gaussian_kernel_2d(s, s, 0.0, ksize)
        k = k ** rng.uniform(0.4, 1.0)
        k = k / k.sum()
    return k


def _conv2d_sym(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Reflect-padded conv with a 2D kernel, keeping shape. x: [C,H,W]."""
    c = x.shape[0]
    kk = k.to(x.device, x.dtype)[None, None].expand(c, 1, -1, -1)
    pad = k.shape[-1] // 2
    x = x[None]
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kk, groups=c)[0]


# ------------------------------------------------------------- operators


def _blur(x: torch.Tensor, sf: int, rng: random.Random) -> torch.Tensor:
    x = _conv2d_sym(x, _random_kernel(sf, rng))
    if rng.random() < 0.25:             # sinc: ringing / overshoot
        k = sinc_kernel(rng.uniform(math.pi / 3, math.pi), rng.choice([9, 11, 13, 15, 17, 19, 21]))
        x = _conv2d_sym(x, k)
    return x


def _resize(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Random up/down resampling with a random interpolation."""
    mode = rng.choice(["area", "bilinear", "bicubic"])
    scale = rng.choice([0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0])
    if scale == 1.0:
        return x
    _, h, w = x.shape
    nh, nw = max(4, int(round(h * scale))), max(4, int(round(w * scale)))
    return F.interpolate(x[None], size=(nh, nw), mode=mode, align_corners=False if mode != "area" else None)[0]


def _noise(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.5:
        sigma = rng.uniform(1.0, 15.0) / 255.0
        if rng.random() < 0.4:          # gray (channel-correlated) noise
            n = torch.randn(1, *x.shape[1:], dtype=x.dtype) * sigma
        else:
            n = torch.randn_like(x) * sigma
        x = x + n
    if rng.random() < 0.05:             # Poisson (sensor) noise
        lam = rng.uniform(0.05, 3.0)
        x = torch.poisson((x.clamp(0, 1) * lam * 255.0).clamp(min=0)) / (lam * 255.0)
    return x


def _jpeg(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.5:
        return x
    q = int(rng.uniform(30, 95))
    img = Image.fromarray((x.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    arr = torch.from_numpy(np.array(out, copy=True)).permute(2, 0, 1).float() / 255.0
    return arr.to(x.device, x.dtype)


# ------------------------------------------------------------- main entry


def bsrgan_degrade(hr: torch.Tensor, sf: int = 2, seed: int | None = None) -> torch.Tensor:
    """HR -> LR with BSRGAN high-order degradation.

    hr: float tensor [3, H, W] in [0, 1], H/W divisible by sf.
    returns LR [3, H/sf, W/sf] in [0, 1].
    """
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    x = hr.clone()

    # fewer rounds at x2: at x4 BSRGAN uses heavier orders, but x2 already
    # loses less information, so piling on rounds just destroys the image.
    order = rng.choices([1, 2, 3], weights=[0.35, 0.45, 0.20] if sf <= 2 else [0.2, 0.5, 0.3])[0]
    ops: Sequence[str] = ["blur", "resize", "noise", "jpeg"]
    for _ in range(order):
        shuffled = list(ops)
        rng.shuffle(shuffled)
        for op in shuffled:
            if op == "blur":
                x = _blur(x, sf, rng)
            elif op == "resize":
                x = _resize(x, rng)
            elif op == "noise":
                x = _noise(x, rng)
            else:
                x = _jpeg(x, rng)

    # final ring/overshoot pass (BSRGAN applies a sinc filter at the end too)
    if rng.random() < 0.6:
        k = sinc_kernel(rng.uniform(math.pi / 3, math.pi), rng.choice([9, 11, 13, 15, 17, 19, 21]))
        x = _conv2d_sym(x, k)

    mode = rng.choice(["area", "bilinear", "bicubic"])
    _, h, w = hr.shape
    x = F.interpolate(x[None], size=(h // sf, w // sf), mode=mode,
                      align_corners=False if mode != "area" else None)[0]
    return x.clamp(0, 1)


# ------------------------------------------------------------- dataset


def find_images(root: str) -> list:
    exts = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    out = []
    for e in exts:
        out.extend(Path(root).rglob(e))
    return sorted(out)


class BSRGANDataset(Dataset):
    """SwinIR/BSRGAN-style pretraining pairs: crop HR, synthesise LR on the fly.

    Returns the same dict as RealSRCropDataset ({"lr", "hr", "scale"}) so
    train_pixel.py can consume either without changes.
    """

    def __init__(self, root: str, lr_patch: int = 64, scale: int = 2,
                 augment: bool = True, limit: int = 0, seed: int = 42):
        # root may be a comma-separated list (e.g. DIV2K + Flickr2K)
        self.files = []
        for r in [x.strip() for x in root.split(",") if x.strip()]:
            self.files.extend(find_images(r))
        if limit and limit < len(self.files):
            r = random.Random(seed)
            self.files = r.sample(self.files, limit)
        if not self.files:
            raise RuntimeError(f"no images under {root}")
        self.ps = lr_patch
        self.sc = scale
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        ps, sc = self.ps, self.sc
        hs = ps * sc
        with Image.open(self.files[idx]) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < hs or h < hs:                       # pad small images up
                im = _I.new("RGB", (max(w, hs), max(h, hs)))
                im.paste(im.crop((0, 0, w, h)))
                w, h = im.size
            top = random.randint(0, h - hs)
            left = random.randint(0, w - hs)
            hr = np.asarray(im.crop((left, top, left + hs, top + hs)), dtype=np.uint8)

        if self.augment:
            if random.random() < 0.5:
                hr = hr[:, ::-1].copy()
            if random.random() < 0.5:
                hr = hr[::-1].copy()
            k = random.randint(0, 3)
            if k:
                hr = np.ascontiguousarray(np.rot90(hr, k))

        hr_t = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1).float() / 255.0
        lr_t = bsrgan_degrade(hr_t, sf=sc)
        return {"lr": lr_t, "hr": hr_t, "scale": sc}
