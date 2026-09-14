# Math Heads A/B — Negative Result

Date: 2026-09-14  
Recipe: ModSwinIR base + **offset-aligned L1** + EMA, RealSR V3 ×2, 12-image tiled Test.

## Results

| ID | Addition | Steps | RGB | Y | Verdict |
|----|----------|-------|-----|---|---------|
| E11 | Align only | 12k | **31.73** | **32.40** | best |
| E13 | + Wiener band FFT on residual | 12k | 31.11 | 31.78 | **worse** |
| E14 | + AmpPhase (amp scale, freeze phase) | ~6k | 29.92 | 30.39 | **worse / incomplete** |
| E15 | + RadialPSF unsharp gate | 8k | 30.71 | 31.41 | **worse** |

## Why (likely)
- Residual is already ~high-freq; **FFT re-gating is redundant** with trunk + Align.
- Unsharp γ is a weaker form of what the trunk already learns.
- Extra FFT graphs caused **OOM/CUDA instability** on 8GB (needed batch=2, no KPN).

## Recommendation
**Ship Align-only as default.** Keep Wiener/AmpPhase/RadialPSF flags for research; do not enable in production recipes.

Checkpoints: `experiments/improve/E13_align_wiener`, `E14_align_ampphase`, `E15_align_radialpsf`.
