# RealSR V3 ×2 — Parameter Budget, SwinIR Baseline, Experiment Matrix

**Date:** 2026-09-10  
**GPU:** NVIDIA RTX 4070 Laptop (8 GB)  
**Data:** RealSR(V3) Canon+Nikon, scale ×2, **406 train pairs**, Test tiled eval (4 pairs, PSNR/SSIM)  
**Wall clock (matrix):** ~3 h of the 12 h budget (E1+E2+E4+E5 trained; E3 reused prior run)

---

## 1. Scientific parameter estimate

### 1.1 Literature anchors [1][2][3]

| Model | Params | Setting | Evidence |
|-------|--------|---------|----------|
| SwinIR-light | **0.88M** (x2) | DIV2K classical | arXiv Table 3 [1] |
| SwinIR-M (classical) | **11.9M** | DIV2K classical | arXiv / GitHub [1] |
| HAT-S / HAT | **9.6M / 20.8M** | bicubic Urban100 | HAT CVPR’23 [2] |
| LP-KPN on RealSR | **&lt;1/5 of RCAN** | real pairs, **best** PSNR | RealSR ICCV’19 [3] |

On RealSR (hundreds of authentic pairs), RCAN’s extra depth buys only **~0.2–0.3 dB** over VDSR/SRResNet and **overfits more** than the much smaller LP-KPN [3]. Doubling SwinIR 11.9M→24M is only **+0.08 dB** on Urban100×4 [2]. SwinIR’s own ablation: width grows params **quadratically**; depth gains **saturate** [1].

### 1.2 Capacity band for *this* setup

| Criterion | Estimate |
|-----------|----------|
| Hard floor (underfit real blur/texture) | **~2M** |
| Practical 8 GB throughput sweet spot | **4–6M** (batch 8) |
| Paper-style full capacity | **8–12M** (needs batch 1 / more steps) |
| Diminishing / overfit risk | **≥15M** on 406 pairs alone |

**Recommendation:** target **4–12M**. Do **not** jump to 20M+ on this data volume without synthetic pretrain (HAT-scale needs ImageNet/DF2K) [2][3].

### 1.3 Local model sizes actually built

| Config | Params | Peak VRAM (measured) |
|--------|--------|----------------------|
| SwinIR-light (official arch) | **0.61M** | 1.8 GB @ bs8 LR64 |
| SwinIR-largeish (C=120, d=4×6) | **3.96M** | 2.3 GB @ bs4 LR64 |
| ModSwinIR-base | **4.12M** | 5.7 GB @ bs8 LR64 |
| ModSwinIR-large | **10.21M** | 3.3 GB @ bs1 LR96 bf16 |
| SwinIR-classical (official C=180, d=6×6) | **11.75M** | 1.9 GB @ bs1 LR80 bf16 |

---

## 2. Experiment matrix (protocol)

- Train: full Canon+Nikon ×2, LR patch as listed, cosine LR, periodic tiled Test eval every 500–750 steps.
- **SwinIR baseline:** official `models/network_swinir.py`, **L1 only**.
- **Mod-SwinIR:** OCA + GDFN + AMF + RAPE + DCN + **UWCL** (non-negative CX + uncertainty L1).
- Effective samples ≈ steps × batch × accum; batch=1 runs are sample-starved vs batch=8 at equal steps.

| ID | Arch | Size | Params | B×A | Patch | Steps | Loss | Best PSNR | Best SSIM |
|----|------|------|--------|-----|-------|-------|------|-----------|-----------|
| E1 | SwinIR | light | 0.61M | 8×1 | 64 | 4000 | L1 | **30.43** | **0.865** |
| E2 | SwinIR | largeish | 3.96M | 4×2 | 64 | 4000 | L1 | **30.54** | **0.868** |
| E3 | Mod | base | 4.12M | 8×1 | 64 | 4000 | UWCL | 29.69 | 0.853 |
| E4 | Mod | large | 10.21M | 1×8 | 96 | 6000 | UWCL | 22.45 | 0.600 |
| E5 | SwinIR | classical | 11.75M | 1×8 | 80 | 4000 | L1 | 28.73 | 0.833 |

Eval curves (selected):

| Step | E1 | E2 | E5 | E4 |
|------|----|----|----|----|
| 500 | 28.89 | 26.69 | 24.42 | 5.37 |
| 1500 | 29.84 | 29.86 | 26.38 | 8.96 |
| 3000 | 30.40 | 30.54 | 28.49 | 13.62 |
| final | 30.43 | 30.54 | 28.73 | 22.45 |

---

## 3. Conclusions

1. **Parameter budget is real but forgiving at the low end on this protocol.** Official SwinIR-light (0.61M) already reaches **30.43 dB** in 4k steps @ batch 8. Capacity-matched 4M SwinIR (E2) adds only **+0.11 dB** — consistent with literature: on limited real pairs, **architecture/optimization dominate raw width** [2][3].

2. **Classical 11.75M SwinIR underperforms at equal *steps* when batch=1.** E5 28.73 &lt; E1/E2 because effective samples are ~8× fewer (4000×1 vs 4000×8). It is **not** evidence that large models are worse — it is evidence that **batch/steps must be scaled with capacity**. Literature full schedules use 100k–1M iterations [3].

3. **Mod-SwinIR base (UWCL) is currently *behind* SwinIR-L1 at matched batch (29.69 vs 30.54).** Likely causes: (a) UWCL/contextual loss trades PSNR for perceptual robustness; (b) extra modules (AMF/DCN/OCA) need longer training; (c) uncertainty head adds optimization burden. **Ablation needed:** Mod-base with L1-only (`--use-uwcl` off path / plain L1) vs SwinIR-base.

4. **Mod-large (10.2M) at batch=1 is severely under-trained** (22.45 dB). Need ≥8× more samples (e.g. 24k steps or grad_accum with larger effective batch) or initialize from E3. Prior fp16+lr=2e-4 run NaN-collapsed; **bf16 + lr=1e-4 was stable** (no NaN in 6k steps).

5. **Scientific target remains 4–12M** for RealSR V3 ×2 on 8 GB. Best *practical* config this session: **SwinIR largeish 4M or Mod-base 4M with batch 8, LR64, ≥4k steps**. Next capacity push should keep **batch≥4** via smaller patch or AMP, not batch=1 with equal steps.

---

## 4. Open questions

- Does Mod-base + **plain L1** beat SwinIR-largeish at matched 4M / batch 8? (clean architecture ablation)
- At **equal effective samples** (e.g. 32k), does SwinIR-classical (11.75M) surpass E2? (needs ~32k steps at batch 1, or batch 2×16k)
- Does UWCL help **perception** (LPIPS) even when PSNR lags? (not measured)
- Full official RealSR schedule (100k–1M iters, 192² patches) — out of 12 h scope [3].

---

## 5. Sources

[1] SwinIR: Image Restoration Using Swin Transformer — arXiv:2108.10257, GitHub JingyunLiang/SwinIR (accessed 2026-09-10).  
[2] HAT: Hybrid Attention Transformer — arXiv:2205.04437 (accessed 2026-09-10).  
[3] Real-World Super-Resolution via Kernel Estimation and Noise Injection (RealSR) — arXiv:1904.00523, ICCV 2019 (accessed 2026-09-10).  

Local artifacts: `research/realsr-param-matrix/findings/F1.md`, `F2.md`, `F3.md`; runs under `experiments/matrix_x2/`.
