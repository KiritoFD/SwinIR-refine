# Research Brief: Optical / Physical Essence of RealSR — What Truly Helps

**Date:** 2026-09-13  
**Depth:** standard (3–5 angles, 1 follow-up round)  
**Audience:** research engineers training Mod-SwinIR on RealSR V3 ×2 (406 pairs, RTX 4070 8GB).

## Question
From optics and inverse-problem theory — not from stacking modules — what interventions actually raise PSNR/SSIM (and robustness) on **real camera** SR like RealSR, given we already have: residual bicubic head, L1(+light amp/HF), EMA, ~4M–12M transformer trunk, 6k–15k steps?

## Why this brief
Empirical ablation at 6k showed loss-term tweaks (patch-amp, conf-HF, EMA) are ≤0.03 dB — noise. UWCL hurts. Need **first-principles** levers, not more FFN variants.

## Scope
- In: optical forward model, sampling/aliasing, PSF field variation, true treatment of misregistration, noise statistics, what published RealSR-track papers actually gain from, compute-matched interventions.
- Out: generative/perceptual-only tracks as primary; DIV2K-only tricks without RealSR transfer evidence.

## Assumptions
- Stay on RealSR V3 ×2 primary; 8GB GPU; prefer 1–2 high-confidence methods over a matrix.
- “Truly helps” = ≥+0.3 dB on full Test or a clear failure-mode fix with theoretical necessity.

## Angles
1. **F1** — Optical forward model: what is (not) invertible; spatially varying PSF; what the inverse must estimate.
2. **F2** — Misregistration: why pixel losses fail; methods that *solve* alignment (not just reweight loss) with evidence on real pairs.
3. **F3** — What actually moves RealSR numbers in literature 2019–2025 (ablation-backed, real dataset).
4. **F4** — Noise + demosaic + gamma / raw vs sRGB: processing pipeline mismatches.

## Decision after research
Pick **one** primary method + **one** backup; then compose-next implement with a single verification run.
