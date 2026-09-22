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


def dwt_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, levels: int = 2,
                      weight: float = 1.0) -> torch.Tensor:
    """L1 on the wavelet high-frequency subbands (HL/LH/HH) over `levels` scales.

    The LL (low-frequency) band is deliberately NOT penalised here: it is already
    covered by the main pixel L1, and re-adding it would just rescale that term.
    Emphasising the detail bands is the operational way to fight L1's
    mean-shrinkage over-smoothing while keeping the fidelity metric honest.
    """
    loss = pred.new_tensor(0.0)
    p, t = pred, target
    done = 0
    for _ in range(levels):
        if p.shape[-1] % 2 or p.shape[-2] % 2 or p.shape[-1] < 4:
            break
        dp, dt = haar_dwt2(p), haar_dwt2(t)
        C = dp.shape[1] // 4
        loss = loss + F.l1_loss(dp[:, C:], dt[:, C:])
        p, t = dp[:, :C], dt[:, :C]
        done += 1
    if done == 0:
        return pred.new_tensor(0.0)
    return weight * loss / done
