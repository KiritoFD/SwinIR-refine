"""Bijective Haar wavelet (DWT) primitives -- no external dependency.

Why wavelets and not FFT
------------------------
The project already measured frequency-domain *heads* (Wiener / AmpPhase /
RadialPSF) as negative, and those failed for two reasons a wavelet avoids: FFT
is unstable / ill-conditioned as a learned band gain, and its global basis
functions are non-local so they don't respect edges.  The Haar discrete wavelet
transform is:

  * **exactly orthogonal** -- DWT^T DWT = I, so the L1 in the wavelet domain is
    a lossless redistribution of the pixel L1 across frequency subbands, not a
    heuristic reweighting that can drift.
  * a **bijection**: the 2x2 transform maps (2C, H, W) <-> (4C, H/2, W/2) with no
    information lost (it is just an orthogonal change of basis on each 2x2 block).
  * **local**: HL/LH/HH are per-block horizontal / vertical / diagonal detail, so
    emphasising them sharpens edges without the ringing a global band gain gives.

The three high-frequency coefficients (HL/LH/HH) are exactly the subband a plain
L1 regression under-fits (the mean-shrinkage / over-smoothing failure), which is
what the wavelet multi-scale loss re-weights.

Implemented with fixed convolutions (stride 2 for the forward, transposed conv
for the inverse) so autograd flows cleanly and it runs under bf16 autocast.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# 2D orthonormal Haar = outer product of two 1D Haar (1/sqrt2)^2 = 1/2, so every
# subband coefficient is (±a±b±c±d) * _HALF.  The transform matrix M = (1/2) H
# (H the 4x4 Hadamard) is symmetric and orthonormal: M M^T = I, hence M^-1 = M.
_HALF = 0.5


def haar_dwt2(x: torch.Tensor) -> torch.Tensor:
    """One level of orthogonal 2D Haar transform.

    (B, C, H, W) -> (B, 4C, H//2, W//2) with subbands ordered [LL, HL, LH, HH]
    along the channel axis (blocks of C).  Requires even H and W.

      LL = (a+b+c+d)/2   (low frequency / average)
      HL = (a-b+c-d)/2   (vertical edge: left-right difference)
      LH = (a+b-c-d)/2   (horizontal edge: top-bottom difference)
      HH = (a-b-c+d)/2   (diagonal detail)
    where a=x00, b=x01, c=x10, d=x11 for each non-overlapping 2x2 block.
    """
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * _HALF
    hl = (a - b + c - d) * _HALF
    lh = (a + b - c - d) * _HALF
    hh = (a - b - c + d) * _HALF
    return torch.cat([ll, hl, lh, hh], dim=1)


def haar_iwt2(y: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`haar_dwt2` (orthogonal, so it is the exact transpose).

    (B, 4C, H//2, W//2) -> (B, C, H, W)."""
    C = y.shape[1] // 4
    ll, hl, lh, hh = y[:, :C], y[:, C:2 * C], y[:, 2 * C:3 * C], y[:, 3 * C:]
    s = _HALF
    a = (ll + hl + lh + hh) * s
    b = (ll - hl + lh - hh) * s
    c = (ll + hl - lh - hh) * s
    d = (ll - hl - lh + hh) * s
    out = y.new_empty(y.shape[0], C, ll.shape[2] * 2, ll.shape[3] * 2)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out


# ---- general orthonormal FIR wavelets (db2/db4) ------------------------------
# Haar is the crudest basis: it is shift-SENSITIVE (a 1-px shift scrambles the
# subbands) and has poor directional/frequency selectivity.  Higher-order Daubechies
# filters keep orthonormality (so the subband L1 is still a lossless redistribution
# of the pixel L1) while being smoother and better localised -- a cheap way to get a
# "sharper" frequency split.  Decoding low-passes from pywt; high-pass is the QMF
# quadrature mirror hi[n] = (-1)^n lo[L-1-n].
_SQRT2 = math.sqrt(2.0)
_FILTERS = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db2": [(1 + math.sqrt(3)) / (4 * _SQRT2), (3 + math.sqrt(3)) / (4 * _SQRT2),
            (3 - math.sqrt(3)) / (4 * _SQRT2), (1 - math.sqrt(3)) / (4 * _SQRT2)],
    "db4": [-0.07576571478927333, -0.02963552764599851, 0.49761866763344383,
            0.8037387518059161, 0.29785779560527736, -0.09921954357684724,
            -0.012603967262037833, 0.032223100604049004],
}


def _qmf(lo: torch.Tensor) -> torch.Tensor:
    # high-pass quadrature mirror: hi[n] = (-1)^n * lo[L-1-n]
    L = lo.shape[0]
    n = torch.arange(L, device=lo.device, dtype=lo.dtype)
    signs = torch.where(n % 2 == 0, 1.0, -1.0).to(lo.dtype)
    return signs * torch.flip(lo, dims=(0,))


def _axis_hf(x, lo, hi, vertical: bool):
    """Per-channel (depthwise) 1-tap-pair analysis along one axis, stride 2, periodic pad."""
    C = x.shape[1]
    L = lo.shape[0]
    padl, padr = L // 2 - 1, L - 1 - (L // 2 - 1)
    if vertical:
        x = F.pad(x, (0, 0, padl, padr), mode="circular")
        klo = lo.view(1, 1, L, 1).repeat(C, 1, 1, 1)
        khi = hi.view(1, 1, L, 1).repeat(C, 1, 1, 1)
        low = F.conv2d(x, klo, stride=(2, 1), groups=C)
        high = F.conv2d(x, khi, stride=(2, 1), groups=C)
    else:
        x = F.pad(x, (padl, padr, 0, 0), mode="circular")
        klo = lo.view(1, 1, 1, L).repeat(C, 1, 1, 1)
        khi = hi.view(1, 1, 1, L).repeat(C, 1, 1, 1)
        low = F.conv2d(x, klo, stride=(1, 2), groups=C)
        high = F.conv2d(x, khi, stride=(1, 2), groups=C)
    return low, high


def dwt2_fir(x: torch.Tensor, basis: str = "db2") -> torch.Tensor:
    """1-level orthonormal FIR 2D DWT -> (B,4C,H/2,W/2), band order [LL,HL,LH,HH]."""
    lo = torch.tensor(_FILTERS[basis], device=x.device, dtype=torch.float32)
    hi = _qmf(lo)
    x = x.float()
    hlow, hhigh = _axis_hf(x, lo, hi, vertical=False)   # split along width
    ll, lh = _axis_hf(hlow, lo, hi, vertical=True)      # low-width band, split along height
    hl, hh = _axis_hf(hhigh, lo, hi, vertical=True)     # high-width band, split along height
    return torch.cat([ll, hl, lh, hh], dim=1)


def dwt_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, levels: int = 2,
                      weight: float = 1.0, basis: str = "haar",
                      band_w=(1.0, 1.0, 1.0), ll_w: float = 0.0,
                      level_weights=None) -> torch.Tensor:
    """L1 on the wavelet high-frequency subbands (HL/LH/HH) over `levels` scales.

    basis='haar' uses the exact slicing transform (and reproduces the previous loss
    bit-for-bit when band_w=(1,1,1), ll_w=0); 'db2'/'db4' use the orthonormal FIR path.
    `band_w=(w_hl,w_lh,w_hh)` lets the three detail directions be weighted differently
    -- horizontal (LH) vs vertical (HL) edge energy is exactly the lens-astigmatism
    signal; `ll_w>0` re-adds a (small) low-frequency term.  `level_weights` (len=levels)
    gives each SCALE its own weight (coarse detail vs fine detail want different emphasis);
    None = uniform.  The per-level result is normalised by the weight sum so a heavier
    anisotropy does not silently scale the gradient magnitude.
    """
    fwd = haar_dwt2 if basis == "haar" else (lambda z: dwt2_fir(z, basis))
    if level_weights is None:
        level_weights = [1.0] * levels
    w = torch.tensor(list(band_w) + ([ll_w] if ll_w > 0 else []),
                     device=pred.device, dtype=torch.float32)
    denom = float(w.sum().clamp(min=1e-6)) if w.numel() else 1.0
    loss = pred.new_tensor(0.0)
    p, t = pred, target
    done = 0
    for lvl in range(levels):
        h = p.shape[-1]
        v = p.shape[-2]
        if h % 2 or v % 2 or h < 8 or v < 8:
            break
        lw = float(level_weights[lvl]) if lvl < len(level_weights) else float(level_weights[-1])
        dp, dt = fwd(p), fwd(t)
        C = dp.shape[1] // 4
        bands = [(C, 2 * C, band_w[0]), (2 * C, 3 * C, band_w[1]), (3 * C, 4 * C, band_w[2])]
        for a, b, wv in bands:
            loss = loss + lw * float(wv) * F.l1_loss(dp[:, a:b], dt[:, a:b])
        if ll_w > 0:
            loss = loss + lw * float(ll_w) * F.l1_loss(dp[:, :C], dt[:, :C])
        p, t = dp[:, :C], dt[:, :C]
        done += 1
    if done == 0:
        return pred.new_tensor(0.0)
    return weight * loss / (done * denom)


def dwt_hf_loss_shift(pred: torch.Tensor, target: torch.Tensor, n_shift: int = 1,
                      **kw) -> torch.Tensor:
    """Shift-ensemble wavelet-HF loss -- the cheap stand-in for DTCWT's shift
    invariance.  Haar subbands are shift-SENSITIVE (a 1-px shift redistributes
    energy across HL/LH/HH), so penalising one decomposition lets the net satisfy
    the L1 by putting detail in the wrong subband.  Averaging the loss over the 4
    dyadic offsets {(0,0),(1,0),(0,1),(1,1)} (circular roll) makes the target
    shift-invariant, so the model must place detail where it is actually needed.
    n_shift=1 -> exactly dwt_highfreq_loss (back-compatible)."""
    offs = [(0, 0), (1, 0), (0, 1), (1, 1)][:max(1, int(n_shift))]
    if len(offs) == 1:
        return dwt_highfreq_loss(pred, target, **kw)
    tot = pred.new_tensor(0.0)
    for dy, dx in offs:
        p = torch.roll(pred, shifts=(dy, dx), dims=(-2, -1)) if (dy or dx) else pred
        t = torch.roll(target, shifts=(dy, dx), dims=(-2, -1)) if (dy or dx) else target
        tot = tot + dwt_highfreq_loss(p, t, **kw)
    return tot / len(offs)


def dtcwt_hf_loss(pred: torch.Tensor, target: torch.Tensor, levels: int = 2,
                  weight: float = 1.0, tree_basis: str = "db2") -> torch.Tensor:
    """Dual-tree complex wavelet (DTCWT-style) high-frequency loss.

    Two analysis trees -- one at offset (0,0), one at a half-sample offset (1,1) --
    are combined per directional band into a complex coefficient (real=tree A, imag=
    tree B) and the loss is L1 on the MAGNITUDE.  This gives the two properties Haar
    lacks and the plain shift-ensemble only partly recovers: approximate SHIFT
    INVARIANCE (a shift rotates the complex coefficient, barely moving |.|) and
    sharper DIRECTIONAL selectivity.  tree_basis is the (orthonormal) FIR used in
    each tree; db2 by default.  Runs in fp32.
    """
    fwd = (lambda z: haar_dwt2(z)) if tree_basis == "haar" else (lambda z: dwt2_fir(z, tree_basis))
    loss = pred.new_tensor(0.0)
    p, t = pred.float(), target.float()
    done = 0
    for _ in range(levels):
        h, w = p.shape[-1], p.shape[-2]
        if h % 2 or w % 2 or h < 8 or w < 8:
            break
        ap, at = fwd(p), fwd(t)
        bp = torch.roll(fwd(torch.roll(p, shifts=(1, 1), dims=(-2, -1))),
                        shifts=(-1, -1), dims=(-2, -1))
        bt = torch.roll(fwd(torch.roll(t, shifts=(1, 1), dims=(-2, -1))),
                        shifts=(-1, -1), dims=(-2, -1))
        C = ap.shape[1] // 4
        hp, ht, bq, btq = ap[:, C:], at[:, C:], bp[:, C:], bt[:, C:]
        magp = torch.sqrt(hp * hp + bq * bq + 1e-8)
        magt = torch.sqrt(ht * ht + btq * btq + 1e-8)
        loss = loss + F.l1_loss(magp, magt)
        p, t = ap[:, :C], at[:, :C]
        done += 1
    if done == 0:
        return pred.new_tensor(0.0)
    return weight * loss / done
