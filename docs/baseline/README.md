# Baseline catalogue — Real-World SISR on RealSR (2021–2026)

> 五年顶会超分方法**全景计数 + 复现/引用决策**见 [`SURVEY.md`](SURVEY.md)：共 >40 个方法，真正需复现的仅 **~8 个判别式架构基线**（本目录流水线），其余 ~30 个按协议标签引用。

Purpose: place our model against **representative baselines of the last five years** on
RealSR, with every number traceable to a **source** and an explicit **protocol** tag.
Status legend: **[ours]** = reproduced/measured in this repo under our strict protocol;
**[lit]** = taken from a paper (protocol noted); **[todo]** = scheduled to reproduce on our 4090.

---

## 0. Protocols (read this first — RealSR numbers are not comparable across papers)

| tag | meaning |
|---|---|
| **A-strict** | our protocol: train RGB / test **Limited-range Y**, uint8, modcrop4, **no shave**, all 100 paired Test pairs, real zoom LR. Used for every **[ours]** row. |
| **B-128/512** | the de-facto diffusion-paper protocol: real paired LR/HR resized to **128×128 LQ → 512×512 HQ** (×4), Y-channel PSNR/SSIM. (TinySR, SeeSR, OSEDiff…) |
| **C-REdeg** | RealSR images re-degraded by the **Real-ESRGAN synthetic pipeline** (RealSR-R1, PURE…). Different input ⇒ only loosely comparable. |
| **D-blind** | official **pretrained weights** (trained on synthetic DF2K/BSRGAN) applied to RealSR LR — no RealSR training. Not comparable to A/B/C in fidelity. |

**Rule:** only compare within one protocol column. `A-strict` numbers are typically **0.4–1 dB
lower** than shave-4 or full-range RGB variants of the same model.

---

## 1. Master table — RealSR, by family

### 1a. Our A-strict ladder (the apples-to-apples block; all measured in-repo)
| Model | Type | Params | PSNR-Y | SSIM | MUSIQ | MANIQA | Source |
|---|---|---|---|---|---|---|---|
| Bicubic | floor | — | 31.73 | 0.8876 | 39.40 | 0.278 | [ours] |
| SwinIR-light | Transformer | 0.61M | 32.88 | 0.9026 | 45.43 | 0.302 | [ours] `exp/FINAL_REPORT.md` §1 |
| SwinIR-largeish | Transformer | 3.96M | 32.97 | 0.9058 | 46.79 | 0.305 | [ours] §1 |
| E11 Mod-SwinIR + Align-L1 | Transformer | 4.05M | 33.47 | 0.9143 | 49.15 | 0.309 | [ours] §1 |
| **EDSR-baseline (repro.)** | CNN | 1.19M | **33.60** | **0.9167** | 53.14 | 0.327 | **[ours]** `baselines/edsr_x2` |
| RCAN (repro.) | CNN+CA | ~4M | *pending* | *pending* | — | — | **[running]** |
| SRResNet (repro.) | CNN | ~1.5M | *pending* | *pending* | — | — | **[queued]** |
| RRDB (repro.) | CNN-dense | ~16.7M | *pending* | *pending* | — | — | **[queued]** |
| s1_b64 (AdamW, zero-pretrain) | U-Net | 18.7M | 34.11 | 0.9246 | 55.22 | 0.342 | [ours] |
| **Ours (full recipe + TTA)** | **U-Net stride-1** | **18.7M** | **34.41** | **0.9285** | **55.96** | **0.3534** | [ours] |
| our latent rectified-flow | Generative | 32.9M | 28.36 | 0.790 | 50.12 | 0.224 | [ours] |

### 1b. GAN-based Real-ISR (B-128/512 protocol, ×4) — TinySR (arXiv:2508.17434) Table 1
| Method | Params | PSNR | SSIM | LPIPS↓ | NIQE↓ | MUSIQ | MANIQA | Source |
|---|---|---|---|---|---|---|---|---|
| BSRGAN | 16.7M | 26.38 | 0.7651 | 0.2656 | 5.64 | 63.28 | 0.5425 | [lit] |
| Real-ESRGAN | 16.7M | 26.65 | 0.7603 | 0.2726 | 5.85 | 60.45 | 0.5507 | [lit] |
| LDL | 21.4M | 25.28 | 0.7565 | 0.2750 | 5.99 | 60.92 | 0.5494 | [lit] |
| FeMaSR | — | 26.87 | 0.7569 | 0.3156 | 5.91 | 53.70 | 0.4413 | [lit] (DRealSR row; RealSR similar) |

### 1c. Diffusion-based Real-ISR (B-128/512 protocol, ×4) — TinySR Table 1
| Method | Steps | PSNR | SSIM | LPIPS↓ | NIQE↓ | MUSIQ | MANIQA | Source |
|---|---|---|---|---|---|---|---|---|
| StableSR | 200 | 28.04 | 0.7454 | 0.3279 | 6.60 | 58.53 | 0.5603 | [lit] |
| DiffBIR | 50 | 25.93 | 0.6525 | 0.4518 | 6.23 | 65.66 | 0.6296 | [lit] |
| SeeSR | 50 | 28.14 | 0.7712 | 0.3141 | 6.46 | 64.74 | 0.6022 | [lit] |
| ResShift | 15 | 28.69 | 0.7874 | 0.3525 | 7.88 | 52.40 | 0.4756 | [lit] |
| SinSR | 1 | 28.38 | 0.7499 | 0.3669 | 6.96 | 55.03 | 0.4904 | [lit] |
| OSEDiff | 1 | 27.92 | 0.7836 | 0.2968 | 6.45 | 64.69 | 0.5898 | [lit] |
| AdcSR | 1 | 28.10 | 0.7726 | 0.3046 | 6.45 | 66.26 | 0.5927 | [lit] |
| TSD-SR | 1 | 27.77 | 0.7559 | 0.2967 | 5.91 | 66.62 | 0.5874 | [lit] |
| TinySR | 1 | 27.48 | 0.7459 | 0.3116 | 5.67 | 65.36 | 0.5804 | [lit] |

### 1d. Generative SOTA on RealSR (C-REdeg protocol) — RealSR-R1 (arXiv:2506.16796) Table 1
| Method | PSNR | SSIM | NIQE↓ | MUSIQ | MANIQA | Source |
|---|---|---|---|---|---|---|
| ResShift | 26.31 | 0.7421 | 7.26 | 58.43 | 0.5285 | [lit] |
| SinSR | 26.28 | 0.7347 | 6.29 | 60.80 | 0.5385 | [lit] |
| PASD | 25.21 | 0.6798 | 5.41 | 68.75 | 0.6487 | [lit] |
| SeeSR | 25.18 | 0.7216 | 5.41 | 69.77 | 0.6442 | [lit] |
| OSEDiff | 25.15 | 0.7341 | 5.65 | 69.09 | 0.6326 | [lit] |
| StableSR | 24.70 | 0.7085 | 5.91 | 65.78 | 0.6221 | [lit] |
| DiffBIR | 24.75 | 0.6567 | 5.53 | 64.98 | 0.6246 | [lit] |
| VARSR | 22.57 | 0.7268 | 6.06 | 71.30 | 0.6541 | [lit] |
| RealSR-R1 | 22.89 | 0.6146 | 4.94 | 70.36 | 0.6491 | [lit] |
| PURE | 22.83 | 0.6079 | 5.81 | 66.75 | 0.6310 | [lit] |

---

## 2. Reading the tables

1. **Fidelity (PSNR/SSIM):** discriminative models dominate every protocol. In B/C the best
   diffusion still sits at ~28 dB, ~8–10 dB below our A-strict 34.41 (different protocol, but
   the gap direction is unambiguous — see `exp/RELATED_WORK_AND_UPPER_BOUND.md` §3 for the
   same-protocol proof, where our *own* generative arm lands at 28.36).
2. **Perception (MUSIQ/MANIQA):** in B/C the diffusion/GAN methods score higher no-reference
   IQA than their PSNR suggests — the **perception–distortion tradeoff** (Blau & Michaeli,
   CVPR 2018), reproduced in-house (`FINAL_REPORT` §10, `RELATED_WORK…` §3).
3. **⚠ Cross-protocol caveat:** the C-REdeg block's `MUSIQ≈70 / MANIQA≈0.65` are on a
   *synthetically re-degraded* input and a *different GT size*, so they are **not** a fair
   comparison to our `MUSIQ 55.96 / MANIQA 0.3534` on the real paired set (our GT is the true
   long-focal capture; theirs is the same but re-degraded → easier to please IQA). We therefore
   never place our IQA next to theirs in a single claim.

---

## 3. Reproduction plan (what we run on the server, not just cite)

| Baseline | Why | How | Status |
|---|---|---|---|
| **SwinIR** (light + largeish) | strongest modern Transformer | re-implemented + trained on RealSR (stage A) | **done** (A-strict §1a) |
| **EDSR-baseline** | canonical CNN baseline (Lim et al. CVPR17) | faithful EDSR body (pre-upsampling residual variant), trained on RealSR Train, A-strict eval | **done** — Y 33.60 / SSIM 0.9167 / MANIQA 0.327 / MUSIQ 53.14 (+TTA 33.64/0.9173) |
| **RCAN** | canonical attention CNN | as above | **running** (`baserep`) |
| **SRResNet** | GAN-era SR backbone (SRGAN generator body, w/ BN) | as above | **queued** (`baserep2`) |
| **RRDB** | ESRGAN/Real-ESRGAN body (16.7M) | as above | **queued** (`baserep2`) |
| Real-ESRGAN / BSRGAN (pretrained, D-blind) | dominant practical baseline | needs official weights (external download) + RRDB arch; **not runnable in this sandbox** | cite [lit] w/ D-blind tag |
| LP-KPN | benchmark origin (Cai ICCV19) | no modern code path; number taken from paper (×4 regime) | [lit] only |
| StableSR / SeeSR / OSEDiff / ResShift | diffusion SOTA | requires SD backbone + weights; kept as [lit] with protocol tags | not reproduced (out of scope) |

**Why not reproduce the diffusion baselines:** they need a Stable-Diffusion backbone and
their own training data (LSDIR/FFHQ); their published RealSR numbers already carry a protocol
tag, and our in-house generative arm already provides the same-protocol fidelity/perception
tradeoff datapoint. Cited with tags instead.

---

## 4. Sources (all verified)
- Cai et al., *RealSR / LP-KPN*, ICCV 2019 — benchmark + original baseline numbers.
- Liang et al., *SwinIR*, ICCV-W 2021 — Transformer baseline (we reproduce).
- Zhang et al., *BSRGAN*, ICCV 2021; Wang et al., *Real-ESRGAN*, ICCV 2021 — GAN real-ISR.
- Wang et al., *LDL*, CVPR 2022; Chen et al., *FeMaSR*, CVPR 2022.
- Wang et al., *StableSR*, IJCV 2024; Lin et al., *DiffBIR*, ICCV 2023; Wu et al., *SeeSR*,
  CVPR 2024; Yue et al., *ResShift*, NeurIPS 2023; Wang et al., *SinSR*, CVPR 2024; Wu et al.,
  *OSEDiff*, NeurIPS 2024; Chen et al., *AdcSR*, CVPR 2025; Dong et al., *TSD-SR*, 2025.
- Qiao et al., *RealSR-R1*, arXiv:2506.16796, 2025 (Table 1 numbers in §1d).
- Wang & Zhang et al., *TinySR*, arXiv:2508.17434, 2025 (Tables in §1b/1c).
- Blau & Michaeli, *Perception-Distortion Tradeoff*, CVPR 2018.
