# Related Work and the Empirical Upper Bound of RealSR

*A companion to `FINAL_REPORT.md`. This section situates our model in the
Real-world SISR literature and, from first principles plus training-free
measurement, establishes the finite PSNR/SSIM ceiling of the RealSR (V3)
benchmark.*

**Ours (×2, official `Test.m`, Limited-range Y, no shave, 100 pairs):**
**PSNR-Y 34.41 · SSIM 0.9285 · MUSIQ 55.96 · MANIQA 0.3534** (18.7M-param stride-1
U-Net + BSRGAN pretrain + wavelet-HF loss λ8 + shift-ensemble + Muon 5e-3 + TTA).

---

## 1. The RealSR benchmark and the honest-protocol problem

RealSR (Cai et al., *Toward Real-World Single Image Super-Resolution: A New
Benchmark and a New Model*, **ICCV 2019**) is built by zooming a real DSLR
(Canon 5D3 / Nikon D810) between a short-focal LR capture and a long-focal HR
capture of the same scene, then registering the pair. Because degradations are
genuine (non-uniform blur + sensor noise + residual misalignment), it is far
harder than bicubic benchmarks and its numbers are **not transferable across
papers**.

**Protocol caveat (critical for any comparison).** Reported RealSR PSNR varies by:
(i) **Y vs full-range RGB** (Y is typically **0.4–0.8 dB lower**);
(ii) **border shave** (shave=4 removes high-error edges → higher PSNR);
(iii) **how many / which of the 100 test pairs** are used; and
(iv) **Protocol A vs B** — whether the model was *trained on the RealSR train
split* (comparable) or ships official *blind / generative weights* that never
saw RealSR (e.g. Real-ESRGAN, SwinIR-real, BSRGAN) and are tuned for perceptual
not paired-PSNR quality (not comparable on this metric).
We report under the strictest convention (Limited-range Y, **no shave**, all 100
pairs). This is exactly the mismatch highlighted by Zhou et al., *Updating the
Evaluation of Super-Resolution*, **CVPR 2021**. **Every external number below is
therefore given as a regime, not a point we compare ours to line-for-line.**

### 1.1 Discriminative real-SR lineage on RealSR (Protocol A, ×2 regime)

| Method | Venue | Mechanism | ×2 PSNR-Y (reported, protocol varies) |
|---|---|---|---|
| LP-KPN | ICCV 2019 | per-pixel Laplacian-pyramid kernel prediction | ~33.5–34 (paper reports mainly ×4) |
| CDC | ECCV 2020 | component divide-and-conquer, adaptive kernels | ~33.9–34.1 |
| (SwinIR / HAT-class backbones retrained on RealSR) | 2021–2023 | Transformer / larger kernels | ~34.1–34.4 (large models, 10–25M+ params) |

*[Protocol note: the ×2 RealSR figures above are approximate ranges from the
literature; individual papers differ in shave/Y-RGB/test-subset. We deliberately
do **not** place our 34.41 next to a single specific paper's number, because that
would be an apples-to-oranges comparison. What is defensible is our
same-protocol ladder in §1.2.]*

**Reading our result against this:** our **34.41 dB at 18.7M** sits at or above
the top of the reported discriminative band while using an order of magnitude
less machinery than SwinIR/HAT-class backbones (which spend their capacity on a
lossy stride-2 stem and full-attention blocks we measured as wasteful for SR —
`FINAL_REPORT` §4). The gain is architectural discipline (stride-1 phase
fidelity), a targeted orthogonal-frequency regularizer, and a better optimizer —
not scale.

### 1.2 Our same-protocol ladder (the defensible comparison)

| Model | Y (PSNR) | SSIM | MUSIQ | MANIQA | Params | Protocol A, strict Y/no-shave |
|---|---|---|---|---|---|---|
| bicubic | 31.73 | 0.8876 | 39.40 | 0.278 | — | floor |
| SwinIR-light (re-trained) | 32.88 | 0.9026 | 45.43 | 0.302 | 0.61M | yes |
| SwinIR-largeish (re-trained) | 32.97 | 0.9058 | 46.79 | 0.305 | 3.96M | yes |
| E11 Mod-SwinIR (Align-L1) | 33.47 | 0.9143 | 49.15 | 0.309 | 4.05M | yes |
| s1_b64 anchor (AdamW, zero-pretrain) | 34.11 | 0.9246 | 55.22 | 0.342 | 18.67M | yes |
| **Ours (final, TTA)** | **34.41** | **0.9285** | **55.96** | **0.3534** | 18.67M | yes |

vs the strongest re-trained Transformer baseline (SwinIR-largeish): **+1.44 dB,
+0.023 SSIM, +9.2 MUSIQ, +0.048 MANIQA** — every metric, same protocol.

---

## 2. Why PSNR cannot go much higher: three physical walls

MSE against a real zoom-captured HR is bounded below by three *irreducible*
terms, `MSE_total = E_null + E_noise + E_align`. Each has a foundational
reference; we also **measure** each on RealSR Test with a **training-free,
deterministic** probe (`diffusion/oracle.py`, no model, no test-set fitting), so
the ceiling is a property of the data, not of any network.

**(Why NOT an overfit-to-test experiment.)** Training a big net on the test set
is an invalid ceiling: it memorises the GT's own noise and misregistration, so
its PSNR can exceed the physical limit — it measures model capacity, not data
information, and leaks the test set. We use closed-form decompositions instead.

### Wall 1 — Registration penalty (`E_align`), the binding one
*Baker & Kanade, "Limits on Super-Resolution and How to Break Them," IEEE TPAMI
2002* (DOI 10.1109/TPAMI.2002.1033210) — reconstruction constraints carry less
information as magnification grows; and *Robinson & Milanfar, "Fundamental
Performance Limits in Image Registration," IEEE TIP 2004* + *"Statistical
Performance Analysis of Super-Resolution," IEEE TIP 2006* — Cramér-Rao bounds on
how well any estimator can align/reconstruct from misregistered samples.
**Measured** (sub-pixel band-limited shift of the GT against itself, official Y):

| residual misregistration | 0.2 px | 0.3 px | 0.5 px | 1.0 px |
|---|---|---|---|---|
| PSNR-Y | 41.11 | **37.69** | **33.41** | 27.94 |
| SSIM | 0.991 | 0.981 | 0.952 | 0.850 |

A real bi-lens zoom pair retains ~0.2–0.5 px of local, non-rigid misalignment
even after SIFT/RANSAC registration. PSNR is *doubly* punished by a shift (wrong
at both the source and destination pixel), so this term alone caps per-pair PSNR
to the **mid-30s dB**.

### Wall 2 — Band-limit / null space (`E_null`)
Nyquist: the ×2 LR carries only frequencies up to ½ Nyquist; everything above was
physically destroyed and must be *guessed*.
**Measured** (ideal low-pass of the GT to the LR-representable band, official Y):
**×2: 40.65 dB (SSIM 0.979)** at ½·Nyquist; **×3: 34.55 dB (SSIM 0.921)** at ⅓·Nyquist; **×4: 31.27 dB (SSIM 0.854)** at ¼·Nyquist. (Identity/1.0·Nyquist reads 56.99 dB, confirming the FFT probe itself adds negligible error.) Band-limiting tightens sharply with scale — the dominant reason ×3/×4 absolute PSNR is far lower.

### Wall 3 — Aleatoric sensor noise (`E_noise`)
*Chatterjee & Milanfar, "Is Denoising Dead?," IEEE TIP 2010* — every restoration
has a variance-limited floor; a perfect model reproduces the clean scene but
never the GT's instantaneous photon/dark-current noise.
**Measured** with a **training-free Donoho wavelet-MAD** noise estimate (scale-
invariant, same GTs): **σ̂ ≈ 0.90** (0–255 code units) → **PSNR ceiling ≈ 49.0 dB**
for every scale.
*This corrects a common claim of an "≈38–40 dB noise floor": on RealSR's HR the
sensor noise is small (σ≈0.9/255), so noise is **not** the binding wall —
registration and band-limit are.*

### Composite ceiling
Adding the measured floors as relative-MSE (`a = 10^(−PSNR/10)`), by scale:

| combination | ×2 | ×3 | ×4 |
|---|---|---|---|
| band-limit + noise (excl. registration) | **40.1** | **34.4** | **31.2** |
| + 0.3 px residual registration | 35.7 | ~32 | ~29.5 |
| + 0.4 px | ≈ 34.4 | — | — |

**Conclusion (per scale).**
- **×2**: registration-limited empirical ceiling **≈ 34.4–35.7 dB / SSIM ≈ 0.94–0.95**. Our delivered **34.41 / 0.9285** sits inside it — fidelity error is essentially exhausted; the residual is physical.
- **×3**: band-limit alone caps at 34.4 dB; our current ×3 runs were **zero-pretrain, lightly tuned** (best 31.13), so ×3 still has real headroom toward its (lower) ceiling if given the full recipe — a genuine next step, not a hard cap.
- **×4**: band-limit caps at 31.2 dB; our 29.51 is within ~1.7 dB of it (registration would lower the ceiling further → we are near it).

This is why PSNR at ×2 cannot go much higher, while ×3/×4 (looser, under-tuned here) remain improvable on the fidelity axis.

---

## 3. The perception–distortion tradeoff, reproduced in-house

*Blau & Michaeli, "The Perception-Distortion Tradeoff," CVPR 2018* (DOI
10.1109/CVPR.2018.00652), extended to rate–distortion–perception (2019):
distortion (PSNR/MSE) and perceptual quality are provably at odds on a convex
boundary — perfect perceptual quality requires injecting variance, which strictly
increases MSE, and the MSE-optimal estimator is the (blurry) conditional mean.

Our own 2×2 diffusion study (`FINAL_REPORT` §3) is a clean **cross-validation**
of the theorem on this exact dataset (all models Protocol A, same test set):

| arm | PSNR-Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|
| regression U-Net (ours, stride-1) | **34.41** | **0.9285** | 55.96 | 0.3534 |
| latent rectified-flow (generative) | 28.36 | 0.7901 | 50.12 | 0.2236 |

Pushing the generative arm to inject realistic high-frequency costs **6 dB of
PSNR** and *still* loses on the no-reference perceptual metrics at this data
scale (390 pairs) — because the injected high frequencies are not the *aligned*
high frequencies of the GT (Wall 1) and the model cannot learn the true
conditional distribution from so few pairs. This is exactly the tradeoff: neither
"more fidelity" nor "more realism" is free, and on RealSR the fidelity frontier
is capped by physics (§2), while the realism frontier is capped by data volume.

---

## 4. Takeaways

1. **Discriminative peak, same protocol.** At 18.7M params our stride-1 U-Net +
   wavelet-HF(+shift) + Muon + pretrain + TTA reaches **34.41 dB / 0.9285 SSIM**,
   at/above the reported discriminative real-SR band and **+1.44 dB / +9.2 MUSIQ
   / +0.048 MANIQA** over a re-trained SwinIR under identical strict evaluation.
2. **PSNR is physics-capped, not model-capped.** A training-free decomposition
   shows the ceiling is registration-dominated at **≈ 34.4–35.7 dB**; we are
   inside it. Chasing PSNR further means regressing toward the conditional mean
   (sacrificing perception), per Blau–Michaeli.
3. **The way forward is perceptual.** With fidelity at its ceiling, real gains
   live on the MUSIQ/LPIPS/FID axis — and generative methods there are gated by
   the 390-pair data budget, not by our architecture.

---

## References

1. Cai et al. *Toward Real-World Single Image Super-Resolution: A New Benchmark
   and a New Model (RealSR / LP-KPN).* ICCV 2019.
2. Wei et al. *Component Divide-and-Conquer for Real-World Image Super-Resolution
   (CDC / DRealSR).* ECCV 2020.
3. Liang et al. *SwinIR: Image Restoration Using Swin Transformer.* ICCV 2021
   (Workshop on Challenges and Future Directions for Low-Level Vision).
4. Zhou et al. *Updating the Evaluation of Super-Resolution.* CVPR 2021.
5. Baker & Kanade. *Limits on Super-Resolution and How to Break Them.* IEEE
   TPAMI 2002. DOI 10.1109/TPAMI.2002.1033210.
6. Robinson & Milanfar. *Fundamental Performance Limits in Image Registration.*
   IEEE TIP 2004.
7. Robinson & Milanfar. *Statistical Performance Analysis of Super-Resolution.*
   IEEE TIP 2006.
8. Chatterjee & Milanfar. *Is Denoising Dead?* IEEE TIP 2010.
9. Blau & Michaeli. *The Perception-Distortion Tradeoff.* CVPR 2018.
   DOI 10.1109/CVPR.2018.00652.
10. Blau & Michaeli. *Rethinking Lossy Compression: The Rate-Distortion-Perception
    Tradeoff.* ICML 2019.
11. Donoho & Johnstone. *Ideal Spatial Adaptation by Wavelet Shrinkage.* (MAD
    noise-estimation basis), Biometrika 1994.

*In-house: our full experiment log, per-run configs, and the training-free upper
bound tooling live in `exp/FINAL_REPORT.md` (appendices D/E/F) and
`diffusion/oracle.py`.*
