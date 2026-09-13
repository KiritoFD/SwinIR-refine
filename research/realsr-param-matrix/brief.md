# Research Brief: RealSR V3 x2 SR parameter budget + SwinIR baseline matrix

## Question
How many parameters are scientifically justified for image super-resolution on RealSR V3 (real camera degradation, ~400 train pairs/scale/camera) under an 8GB laptop GPU, and does a modernized SwinIR (OCA+GDFN+AMF+RAPE+DCN+UWCL) beat the official SwinIR baseline at matched capacity?

## Scope
- In: published model sizes & metrics for SwinIR/HAT/RealSR-track methods; capacity vs data-size for real-world SR; RealSR V3 protocol; ablation-practice for capacity-matched comparisons.
- Out: new architecture search; multi-scale joint training design; deployment.
- Depth: standard (literature + local empirical matrix).
- Date: 2026-09-10. GPU: RTX 4070 Laptop 8GB. Data: RealSR(V3) Canon+Nikon scale2, 406 train pairs.

## Assumptions
- x2 only for the matrix; tiled eval on 4 Test pairs (PSNR/SSIM).
- Fair protocol: same data/schedule family; SwinIR baseline uses L1; Mod uses UWCL.
- 12h wall-clock budget for the empirical matrix.

## Angles
1. F1 — Published parameter counts & reported metrics (SwinIR light/classical, HAT, RealSR papers).
2. F2 — Capacity vs data regime for real-world SR (when bigger models stop helping).
3. F3 — RealSR V3 dataset properties & common training hyperparams in public code/papers.

## Local empirical track (runs in parallel)
- E1 SwinIR-light 0.61M L1
- E2 SwinIR-largeish 3.96M L1 (capacity-matched to Mod-base)
- E3 ModSwinIR-base 4.12M UWCL (prior run)
- E4 ModSwinIR-large 10.21M batch1 bf16
- E5 SwinIR-classical 11.75M L1 batch1
