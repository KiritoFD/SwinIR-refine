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

**1a-1 — OFFICIAL architecture ladder.** Each model below is the paper's own network, loaded
from `third_party/<repo>` at the published ×2 config (`diffusion/baselines_official.py`, no
re-implementation), trained **from scratch on RealSR Train (390 pairs)** under one shared
recipe: AdamW 1e-4, L1, EMA 0.999, warmup 500, and batch × steps = **1.28M samples** = our
champion's budget. Evaluated with tile 64 / pad 16 / 100 pairs. `unet_ctrl` is *our* U-Net run
under that same recipe, so an architecture claim cannot secretly be an optimiser claim.

| Model (official config) | Type | Params (measured) | batch × steps | PSNR-Y | SSIM | MUSIQ | MANIQA | Status |
|---|---|---|---|---|---|---|---|---|
| VDSR (20 conv, C=64) | CNN | 0.59M | 32 × 40k | 33.6205 | 0.91675 | 52.91 | 0.31747 | done — +TTA 33.6780 / 0.91755 / 52.85 / 0.31992 |
| **EDSR-baseline** (16 RB × 64F, res_scale .1, sub-pixel) | CNN | **1.37M** | 32 × 40k | 33.4946 | 0.91489 | 51.49 | 0.32079 | done — +TTA 33.5300 / 0.91549 / 51.49 / 0.32231 |
| RDN-B (D=16, C=8, G=64, G0=64) | CNN-dense | 22.12M | 32 × 40k | 33.9061 | 0.92179 | 54.83 | 0.33954 | done — +TTA 33.9810 / 0.92266 / 54.74 / 0.34090 |
| RCAN (G=10, R=20, F=64, r=16) | CNN+CA | 15.44M | 64 × 20k | 33.7818 | 0.91987 | 54.03 | 0.33213 | done — +TTA 33.8488 / 0.92081 / 53.95 / 0.33449 |
| SwinIR-L classical (6×6, dim180, win8) | Swin Transformer | 11.75M | 8 × 160k（**实跑 132k 步 = 1.06M 样本**） | 33.9707 | 0.92242 | 54.59 | 0.33782 | done — +TTA 34.0241 / 0.92298 / 54.57 / 0.33873 |
| SRFormer (dim180, 6×6, win24, permuted self-attn) | Transformer | 10.40M | 16 × 80k | 33.8724 | 0.92365 | 54.68 | 0.33907 | done — +TTA 33.9989 / 0.92450 / 54.67 / 0.33984 |
| HAT-L (RHAG=12 × HAB=6, dim180, win16, OCA+CAB+CLAF) | Hybrid attention | 40.70M | 4 × 320k（**实跑 50k 步 = 0.20M 样本**，早停） | 33.5457 | 0.91646 | 51.95 | 0.32060 | done — +TTA 33.5931 / 0.91725 / 51.92 / 0.32361 |
| **`unet_ctrl` — our architecture, baseline recipe** | U-Net stride-1 | 18.67M | 128 × 10k | **34.0339** | **0.92398** | **54.96** | **0.33991** | done — +TTA **34.1664 / 0.92525 / 54.91 / 0.34147** |
| **Ours (delivered: + DF2K-style pretrain + Muon 5e-3 + wavelet λ8 + shift-ensemble + TTA)** | **U-Net stride-1** | **18.67M** | 128 × 10k (+3.2M-sample pretrain) | **34.41** | **0.9285** | **55.96** | **0.3534** | **done** |
| Bicubic | floor | — | — | 31.73 | 0.8876 | 39.40 | 0.278 | done |

**What the official ladder says (8 of 8 arms measured):**

1. **The architecture claim survives without our extras.** `unet_ctrl` is our 18.67M stride-1 U-Net
   trained with the *baselines'* recipe — AdamW 1e-4, plain L1, no pretraining, no wavelet loss,
   no Muon, no shift-ensemble — and it leads every official arm on all four metrics: **+0.14 dB /
   +0.0023 SSIM / +0.34 MUSIQ over SwinIR-L** (best PSNR among baselines), **+0.17 dB / +0.0008 SSIM /
   +0.24 MUSIQ over SRFormer** (best SSIM among baselines), **+0.19 dB over RDN-B**, **+0.57 dB over
   HAT-L**. So the delivered margin is not an optimiser or pretraining artifact; the recipe stack
   (Muon +λ8 + pretrain + TTA) adds the remaining +0.24 dB / +0.0033 SSIM / +1.05 MUSIQ on top.
2. **Bigger is not better at 390 pairs.** HAT-L (40.7M) lands at 33.5457 — *below* SwinIR-L (11.75M,
   33.9707) and barely above EDSR-baseline (1.37M, 33.4946). Two causes, both honest: HAT's own
   recipe relies on ImageNet + same-task pretraining chains we do not use (no weights obtainable
   here), and its run early-stopped at ~50k of 320k steps under the shared patience rule, i.e. it
   consumed the least data. This reproduces our own capacity-saturation result (18.7M → 72.5M = +0.05 dB).
3. **Fairness caveat, stated once:** every arm shares data, recipe, sample *cap* (batch × steps =
   1.28M) and the same early-stopping rule (patience 20 on a 16-pair val, min 8000 steps), and the
   **actually consumed** sample count is now printed in the batch column of the table above: HAT-L
   stopped at 50k/320k (0.20M samples) and SwinIR-L at 132k/160k (1.06M); the other six arms ran the
   full 1.28M. So "same budget" here means the same *ceiling*, not the same consumption. Strict
   equal-sample runs would need `--patience` disabled per arm; not done, because early stopping is
   also what our own champion arm uses.

Param counts are **trainable parameters, measured from our own checkpoints** (registered buffers
listed separately: SwinIR-L 11.75M + 2.80M buffers, SRFormer 10.40M + 8.96M, HAT-L 40.70M +
0.21M; `audit_stats.sh`). They are not copied from the papers, whose figures are a different model:
EDSR 1.52M / RCAN 16.70M / SwinIR-L 11.88M are the shared ×2+×3+×4-head networks (our SwinIR-L
matches its builder's kwargs verbatim and lands 1.1% below the paper's 11.88M, which we have not
reconciled layer-by-layer), and 20.83M is **HAT**, not HAT-L (HAT-L doubles the depth, RHAG 6 → 12,
as in the official `train_HAT-L_SRx2_*.yml`).

**Are the margins real? (per-pair tests on the 100 pairs, `audit_stats.sh`)**

| comparison | mean Δ PSNR-Y | wins | sign test p | paired t (p) |
|---|---|---|---|---|
| `unet_ctrl` vs SwinIR-L, **single forward** | +0.063 dB | 60/100 | 0.057 | +2.10 (0.035) |
| `unet_ctrl` vs SwinIR-L, **with TTA** | +0.142 dB | 67/100 | 8.7e-4 | +4.69 (2.7e-6) |
| `unet_ctrl` vs SRFormer, single forward | +0.162 dB | 77/100 | 5.5e-8 | +5.64 (1.7e-8) |
| `unet_ctrl` vs SRFormer, with TTA | +0.168 dB | 76/100 | 1.8e-7 | +6.36 (2.0e-10) |
| `unet_ctrl` vs RDN-B, single forward | +0.128 dB | 67/100 | 8.7e-4 | +4.37 (1.2e-5) |
| ours (delivered, +TTA) vs SwinIR-L (+TTA) | +0.382 dB | 87/100 | 1.3e-14 | +10.33 (≈0) |
| ours (delivered, +TTA) vs SRFormer (+TTA) | +0.407 dB | 91/100 | 3.3e-18 | +12.42 (≈0) |
| ours (delivered, +TTA) vs every other official arm | +0.43 … +0.88 dB | 87–93/100 | ≤1.3e-14 | +10.7 … +12.2 |

Two cells are thin and we say so explicitly: **PSNR-wise, our bare architecture at a single forward
is only marginally ahead of SwinIR-L (+0.063 dB, p≈0.04–0.06)**, and **SSIM-wise SRFormer is almost
level with it** (0.92365 vs 0.92398 single forward, a gap of 3.3e-4). With the same TTA treatment as
our headline row the PSNR gap to SwinIR-L becomes solidly significant (+0.142 dB, p≈1e-3) and every
other comparison is ≥ +0.16 dB at p < 1e-3. This is exactly why SSIM/MUSIQ/MANIQA are the primary
criteria here and why a single-metric claim would be fragile.

**Why every mature model lands within ~0.5 dB on this dataset:** the ceiling is set by the data, not
by capacity - our training-free decomposition (`exp/RELATED_WORK_AND_UPPER_BOUND.md`, `diffusion/oracle.py`)
puts the registration-limited achievable PSNR-Y at **34.4–35.7 dB** with bicubic at 31.73, i.e. the whole
usable dynamic range is ~2.7 dB. The published DIV2K ×2 ladder is compressed the same way (EDSR-baseline
37.56 → RCAN 37.67 → SwinIR-L 38.14 → HAT-L 38.34 = 0.78 dB across four generations).

Deliberately **absent** from 1a-1: RLFN (official repo unreachable from this box, and a
re-implementation that misses the paper's 2.11M would be exactly the error this ladder exists
to avoid) and SRCNN (24.7k official; only our variant exists).

**1a-2 — pre-upsampling residual VARIANTS (our harness's design, NOT the official models).**
These take the bicubic-upsampled image and predict a residual. They are kept because the
comparison is itself a finding (see the EDSR row pair), but they must never be quoted as
"EDSR" / "RCAN". Evaluated at tile 64 / **pad 8** (the older margin) — IQA cells are pending a
pad-16 re-evaluation.

| Variant | Params | PSNR-Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|---|
| EDSR variant (pre-upsampling residual) | 1.19M | 33.5953 | 0.91666 | 53.14 | 0.32726 |
| RCAN variant (10 RG × 8 RB) | 3.76M | 33.8274 | 0.92024 | 54.47 | 0.33665 |
| RRDB variant (ESRGAN body) | 16.7M | 33.6583 | 0.91780 | 53.56 | 0.33050 |
| SRResNet variant (BN body) | ~1.5M | 31.4061 | 0.88989 | 40.73 | 0.26553 |

(all at tile 64 / pad 8; +TTA values in the summary of `experiments/diffusion/baselines/*`)

SRResNet's row is a genuine training-pathology record, not a protocol artifact: its VAL peaked
at 29.8 dB @ step 5000 and fell monotonically to 20.9 dB @ 20000 (overfitting of a BN body on
390 pairs at ~2500 epochs per image), so `ckpt_best` came from step 5000 — and its MUSIQ 40.73
sits at the bicubic level (39.40), i.e. the network output had degenerated rather than merely
lost a little fidelity.

**First finding from pairing 1a-1 and 1a-2 (EDSR only so far):** the official post-upsampling
model is *worse* on our paired protocol than the same body refining a bicubic upsample
(33.4946 vs 33.5953 dB, MUSIQ 51.49 vs 53.14). That is the same effect that made our stride-1
HR-space design win (+0.38 dB over stride-2, `FINAL_REPORT` §4) — under bi-lens registration
residuals the network must correct detail **at the target resolution**, and a sub-pixel head
upstream of that correction cannot.

**1a-3 — our own stage-A / exploratory rows** (same evaluator, kept for the chronicle).

| Model | Type | Params | PSNR-Y | SSIM | MUSIQ | MANIQA | Source |
|---|---|---|---|---|---|---|---|
| SwinIR-light (our re-train) | Swin Transformer | 0.61M | 32.88 | 0.9026 | 45.43 | 0.302 | [ours] `FINAL_REPORT` §1 |
| SwinIR-largeish (our re-train) | Swin Transformer | 3.96M | 32.97 | 0.9058 | 46.79 | 0.305 | [ours] §1 |
| E11 Mod-SwinIR + Align-L1 | Swin Transformer | 4.05M | 33.47 | 0.9143 | 49.15 | 0.309 | [ours] §1 |
| s1_b64 (AdamW 3e-4, zero-pretrain) | U-Net stride-1 | 18.67M | 34.11 | 0.9246 | 55.22 | 0.342 | [ours] §4 |
| our latent rectified-flow | Generative | 32.9M | 28.36 | 0.790 | 50.12 | 0.224 | [ours] §3 |


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
| **EDSR / RCAN / RDN / VDSR / SwinIR-L / SRFormer / HAT-L** | the official architectures, so the ladder measures the architecture | official code from `third_party/` via `diffusion/baselines_official.py` (post-upsampling, `--net-input lr`), trained from scratch on RealSR Train, batch × steps = 1.28M, eval A-strict tile 64 / pad 16 + TTA + IQA | **running** (`tmux bl`, `scripts/server/run_official_bl.sh`; batch measured per arm by `probe_batches.sh`) |
| `unet_ctrl` | our own architecture under the baselines' recipe (AdamW 1e-4, plain L1) | same as above, 128 × 10k | running |
| **SwinIR** (light + largeish) | strongest modern Transformer (stage A) | re-implemented + trained on RealSR (stage A) | **done** (A-strict §1a-3) |
| EDSR/RCAN/RRDB/SRResNet + SRCNN/VDSR/RDN/RLFN/SRFormer/HAT **variants** | what our harness's pre-upsampling residual form costs/buys | `diffusion/baselines_sr.py`, same protocol | **done** as §1a-2 variants (labelled; not the official models) |
| RLFN, SRCNN (official) | efficient-track winner / historical first CNN | official code not reachable / params cannot be matched | **excluded**, stated in §1a-1 |
| Real-ESRGAN / BSRGAN (pretrained, D-blind) | dominant practical baseline | needs official weights (download.pytorch.org → 403 on this box) + VGG perceptual weights | cite [lit] w/ D-blind tag |
| LP-KPN | benchmark origin (Cai ICCV19) | no modern code path; number taken from the paper (×4 regime) | [lit] only |
| StableSR / SeeSR / OSEDiff / ResShift / VOSR … | diffusion SOTA | need an SD/FLUX backbone + LSDIR-scale data; their RealSR numbers carry a protocol tag, and our own generative arm gives the same-protocol tradeoff point | cited with tags, not reproduced |

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
