"""Adversarial degradation mining: min-max over a *differentiable* degeneration.

Why
---
390 RealSR pairs trained for 10k steps x batch 96 is ~2460 epochs: the empirical
distribution has collapsed onto a few thousand discrete points, and the
measured ceiling of "more params / more steps / fixed-pipeline pretraining"
is +0.05 / +0.01 / +0.09 dB.  What is missing is degradation DIVERSITY, and
BSRGAN's answer -- a fixed random recipe -- only partially overlaps real camera
degradation (measured transfer: +0.09 dB for 4.8 h).

Instead of sampling degradations from a fixed pipeline, solve the robust
optimisation

    min_theta max_phi  E_x [ L( F_theta( G_phi(x) ), x ) ]

where G_phi turns an HR patch into its *hardest* LR view, online.  The inner
max finds, by gradient ascent, the point of the degradation manifold the
current SR net is worst at; the outer min trains the net on exactly that.
This is active sampling on a continuous manifold around the training images.

The physical constraint set Phi (so the adversary cannot "win" by outputting
black images):
  * blur = a CONVEX COMBINATION of a fixed bank of physically plausible PSFs
    (isotropic + anisotropic Gaussians; softmax weights) applied with a strided
    depthwise conv -- the x2 decimation itself, differentiable in phi
  * additive Gaussian noise with sigma = sigma_max * sigmoid(.) in [0, sigma_max]
JPEG / random-resize are NOT in v1: they are non-differentiable, so they cannot
take gradients in the inner max.  They can be re-added later as extra
non-adversarial random ops composited after G_phi.

Min-max schedule (--adv-inner in train_pixel):
  * 0 (default): simultaneous GDA -- one forward/backward, phi ASCENDS by
    stepping on the flipped gradient while theta DESCENDS on the same graph.
    Same cost as a normal step.
  * N > 0: true alternation -- N extra forwards through the frozen SR net per
    step, each taking a real ascent step with a freshly regenerated LR.
    N+1 forwards per training step.

Everything here runs in fp32 outside the training autocast: G_phi is tiny and
its softmax/einsum want the precision.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ----------------------------------------------------------------- kernel bank


def build_kernel_bank(ksize: int = 15) -> torch.Tensor:
    """Fixed bank of plausible PSFs, each normalised, centred in a ksize x ksize grid.

    Sigma ranges are chosen against the x2 BSRGAN recipe we already calibrated
    (sigma in [0.05, 1.2] at sf=2) and extended upward so the adversary can
    reach *harder*-than-BSRGAN blur: that headroom is the whole point of the
    inner max.  The first entry is near-identity so "barely degraded" stays
    reachable (the adversary will not pick it, but the simplex needs it).
    """
    ks: list[torch.Tensor] = []
    half = (ksize - 1) / 2.0
    ax = torch.arange(ksize, dtype=torch.float32) - half
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")

    def gauss(sx: float, sy: float, theta: float) -> torch.Tensor:
        ct, st = math.cos(theta), math.sin(theta)
        u = ct * xx + st * yy
        v = -st * xx + ct * yy
        k = torch.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        return k / k.sum()

    # isotropic: near-identity .. stronger than BSRGAN-x2's max
    for s in (0.05, 0.4, 0.8, 1.3, 1.8, 2.4):
        ks.append(gauss(s, s, 0.0))
    # anisotropic (astigmatic / motion-like): 3 x 2 aspect x 2 angle
    for s1 in (0.6, 1.2, 2.0):
        for ratio in (0.3, 0.6):
            for theta in (0.0, math.pi / 4):
                ks.append(gauss(s1, s1 * ratio, theta))
    return torch.stack(ks)  # (K, ksize, ksize)


def per_sample_depthwise(x: torch.Tensor, k: torch.Tensor, stride: int, padding: int) -> torch.Tensor:
    """x: (B,C,H,W), k: (B,ksize,ksize) -> depthwise conv with a DIFFERENT kernel
    per sample, differentiable in both x and k.  Grouped-conv trick."""
    b, c, h, w = x.shape
    ks = k.shape[-1]
    xx = x.reshape(1, b * c, h, w)
    kk = k[:, None, None].expand(b, c, 1, ks, ks).reshape(b * c, 1, ks, ks)
    out = F.conv2d(xx, kk, stride=stride, padding=padding, groups=b * c)
    return out.reshape(b, c, out.shape[-2], out.shape[-1])


# ------------------------------------------------------------------- G_phi


class AdvDegradation(nn.Module):
    """G_phi: HR patch -> LR patch, differentiable in phi.

    A small CNN looks at the HR patch and emits (a) softmax weights over the
    PSF bank and (b) a noise level.  Conditioning the adversary on the CONTENT
    is what makes this active sampling: flat-sky regions get attacked with
    noise, textured regions with the blur that destroys exactly their
    frequency band.
    """

    def __init__(self, scale: int = 2, sigma_max: float = 0.12, ksize: int = 15,
                 hidden: int = 64):
        super().__init__()
        self.scale = scale
        self.sigma_max = float(sigma_max)
        bank = build_kernel_bank(ksize)
        self.register_buffer("bank", bank)
        self.n_kernels = int(bank.shape[0])
        # ksize must be odd so the PSF centre sits on an integer pixel: with
        # stride=2 and padding=ksize//2 the output window for LR pixel i is
        # [2i - r, 2i + r], centred EXACTLY on 2i -- no half-pixel phase shift
        # against the bicubic-up the SR net conditions on.
        assert ksize % 2 == 1
        self.feat = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, hidden, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(hidden, self.n_kernels + 1)

    def forward(self, hr: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.head(self.feat(hr))
        w = torch.softmax(logits[:, : self.n_kernels], dim=1)          # (B,K) convex
        k = torch.einsum("bk,kij->bij", w, self.bank)                  # (B,k,k)
        lr = per_sample_depthwise(hr, k, stride=self.scale, padding=self.bank.shape[-1] // 2)
        sigma = self.sigma_max * torch.sigmoid(logits[:, self.n_kernels])
        if eps is None:
            eps = torch.randn_like(lr)
        return (lr + sigma.view(-1, 1, 1, 1) * eps).clamp(0, 1)

    def kernel_report(self, hr: torch.Tensor) -> dict:
        """Diagnostics: which part of Phi the adversary is actually using."""
        with torch.no_grad():
            logits = self.head(self.feat(hr))
            w = torch.softmax(logits[:, : self.n_kernels], dim=1).mean(0)
            sig = self.sigma_max * torch.sigmoid(logits[:, self.n_kernels]).mean()
        return {"top_kernels": torch.topk(w, 3).indices.tolist(),
                "kernel_entropy": float(-(w * (w + 1e-12).log()).sum()),
                "mean_sigma": float(sig)}


# ------------------------------------------------------------------ dataset


def find_images(root: str) -> list:
    exts = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    out = []
    for e in exts:
        out.extend(Path(root).rglob(e))
    return sorted(out)


class HQPatchDataset(Dataset):
    """HR-only crops for on-GPU degradation (the min-max pretraining source).

    Same discovery / caching conventions as BSRGANDataset; returns {"hr", "idx"}
    because the LR half is produced by AdvDegradation on the GPU, not here.
    """

    def __init__(self, root: str, hr_patch: int = 128, augment: bool = True,
                 limit: int = 0, seed: int = 42, decoded_manifest: str | None = None,
                 deterministic: bool = False):
        from .decoded import DecodedStore

        self.store = DecodedStore(decoded_manifest)
        self.files = []
        for r in [x.strip() for x in root.split(",") if x.strip()]:
            self.files.extend(find_images(r))
        if limit and limit < len(self.files):
            self.files = random.Random(seed).sample(self.files, limit)
        if not self.files:
            raise RuntimeError(f"no images under {root}")
        self.ps = hr_patch
        self.augment = augment
        self.deterministic = deterministic
        hits = sum(1 for f in self.files if self.store.has(f))
        print(f"  HQPatchDataset: {len(self.files)} HR images, {hits} from the decoded cache",
              flush=True)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        ps = self.ps
        full = self.store.get(self.files[idx])
        if full is not None:
            h, w = full.shape[:2]
            if h < ps or w < ps:
                pad = np.zeros((max(h, ps), max(w, ps), 3), dtype=np.uint8)
                pad[:h, :w] = full
                full, h, w = pad, pad.shape[0], pad.shape[1]
            if self.deterministic:
                top, left = (h - ps) // 2, (w - ps) // 2
            else:
                top = random.randint(0, h - ps)
                left = random.randint(0, w - ps)
            hr = full[top : top + ps, left : left + ps]
        else:
            with Image.open(self.files[idx]) as im:
                im = im.convert("RGB")
                w, h = im.size
                if h < ps or w < ps:
                    canvas = Image.new("RGB", (max(w, ps), max(h, ps)))
                    canvas.paste(im, (0, 0))
                    im, w, h = canvas, canvas.size[0], canvas.size[1]
                if self.deterministic:
                    top, left = (h - ps) // 2, (w - ps) // 2
                else:
                    top = random.randint(0, h - ps)
                    left = random.randint(0, w - ps)
                hr = np.asarray(im.crop((left, top, left + ps, top + ps)), dtype=np.uint8)
        if self.augment:
            if random.random() < 0.5:
                hr = hr[:, ::-1]
            if random.random() < 0.5:
                hr = hr[::-1]
            k = random.randint(0, 3)
            if k:
                hr = np.ascontiguousarray(np.rot90(hr, k))
        # copy=True: the decoded store hands out non-writable memmap slices and
        # torch.from_numpy refuses to wrap those without a warning
        t = torch.from_numpy(np.array(hr, copy=True)).permute(2, 0, 1).float() / 255.0
        return {"hr": t, "idx": idx}
