---
feature: realsr-math-innovations
status: delivered
updated: 2026-09-14
branch: main
commits: pending-doc
---

# RealSR Mathematical Network Innovations (3-phase)

## Report

**What was built** — Three inference-time heads on the residual path of ModSwinIR (all optional, zero-init → identity):
1. **WienerHead** (E13): radial FFT band gains `exp(tanh(·))` on residual.
2. **AmpPhaseHead** (E14): FFT residual, amplitude scaled by spatially-meaned per-channel gain, phase frozen.
3. **RadialPSFHead** (E15): `res + γ⊙(res − G_σ*res)` unsharp/de-PSF with spatial γ from features.

Trained with proven **offset-aligned L1 + EMA**, batch=2, accum=4, 8–12k steps on RealSR V3 ×2.

**Verification** — 12-image tiled Test (RGB / Y):

| Run | Head | Steps | RGB PSNR | Y PSNR | vs E11 Align-only |
|-----|------|-------|----------|--------|-------------------|
| E11 | (none) | 12k | **31.73** | **32.40** | baseline |
| E13 | Wiener | 12k | 31.11 | 31.78 | −0.62 / −0.62 |
| E14 | AmpPhase | ~6k* | 29.92 | 30.39 | incomplete, worse |
| E15 | RadialPSF | 8k | 30.71 | 31.41 | −1.0 / −1.0 |

\*E14 crashed mid-run (CUDA OOM/access violation on FFT path); numbers from partial ckpt.

**Journey log**
- Wiener/AmpPhase FFT heads caused OOM at batch≥4 and unstable CUDA on this 8GB box; needed batch=2 + no-KPN + no-AMP (RadialPSF).
- Eval must honor `no_kpn` / `use_wiener` from ckpt args or load_state_dict fails.
- **All three math heads hurt PSNR vs Align-only** under equal-ish budget: residual FFT modulation and unsharp gate are redundant/conflicting with the trunk + Align L1.
- **Keep:** Align loss (+0.57 dB) is the only structural change that moved the needle. Do **not** enable Wiener/AmpPhase/RadialPSF by default.

## [S1] Problem
After Align, remaining optical gap suggested MTF/noise modeling; hypothesis was residual Wiener/amp/radial-PSF heads would help.

## [S2] Design
### S2.1 Wiener band gain — **implemented, harmful**
### S2.2 Amp/phase dual branch — **implemented, harmful/incomplete**
### S2.3 Radial PSF field — **implemented, harmful**
### S2.4 Train recipe — Align L1 + EMA; flags `--use-wiener`, `--use-ampphase`, `--use-radialpsf` (all default **off**)

## [S3] Out of Scope
GAN; multi-scale; 20M+.

## Tasks
- [x] T1: Wiener head smoke + train E13
- [x] T2: E13 12-img eval (31.11 / 31.78)
- [x] T3: AmpPhase head + E14 (partial train)
- [x] T4: RadialPSF head + E15 (8k, 30.71 / 31.41)
