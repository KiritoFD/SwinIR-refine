---
feature: realsr-align-kpn
status: delivered
updated: 2026-09-13
branch: main-worktree-g-realsr
commits: (uncommitted working tree on G:\RealSR)
---

# RealSR Align-first then KPN

## Report

**What was built** — Train-time **offset-aligned L1** (mimicked alignment): a small zero-init conv predicts ≤±3 px residual offsets from (SR, HR), HR is warped toward SR, loss is `(1-α)‖sr−W(hr)‖₁ + α‖sr−hr‖₁ + λ‖off‖₁`. Inference graph unchanged. Official **Y-channel** PSNR/SSIM added to eval. Dataset now PIL-crops patches (fixes OOM on large Nikon HR).

**Verification**
- `OffsetAlignedLoss` smoke: finite, backward OK; offsets start at 0.
- E11 train 12k steps, batch8 LR64, EMA: peak VRAM ~4.93 GB.
- 12-image Test (same pairs): **E11 Align RGB 31.73 / Y 32.40** vs **E9 L1+amp RGB 31.16 / Y 31.75** → **+0.57 RGB, +0.65 Y**.

**Journey log**
- CX/UWCL and tiny loss ablations were noise/negative; optical research pointed to \(T_\tau\).
- Vanilla DCN/CX often fail on RealSR; constrained offset + pure-L1 anchor is safer.
- Y vs RGB is ~0.6–0.7 dB here — always report both.

## [S1] Problem
Residual RealSR misregistration makes pixel L1 compare wrong coordinates; loss-term micro-tweaks did not move PSNR.

## [S2] Design
### S2.1 Offset-aligned L1 — **delivered**
- `--align-loss --max-shift 3 --w-pure 0.25 --w-off 0.01`
- Loss-only aligner; zero-init; tanh×max_shift.

### S2.2 Y metrics — **delivered**
- `eval.py` / online eval emit `psnr_y`, `ssim_y` (BT.601).

### S2.3 LP-KPN head — **not implemented this round**
- Align already gave +0.6 dB; KPN remains optional next experiment.

## [S3] Out of Scope
GAN; CX; raw domain; KernelGAN TTA.

## Tasks
- [x] T1: Offset-aligned loss + flag
- [x] T2: Y PSNR/SSIM in eval
- [x] T3: E11 train vs E9 — +0.57 RGB / +0.65 Y on 12-img
- [ ] T4: LP-KPN head (optional next)
