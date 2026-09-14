# Negative results — do not re-enable by default

All evaluated on RealSR V3 ×2, 12-image tiled Test (RGB / Y), Align recipe unless noted.

## Confirmed harmful

| Idea | Result | Notes |
|------|--------|-------|
| **UWCL / CX as main loss** | −0.95 dB vs L1 | CX on RealSR can transfer −2–3 dB; never use as recon loss |
| **Wiener FFT band gain on residual** (E13) | 31.11 / 31.78 vs Align 31.73 / 32.40 | OOM-prone; residual already high-freq |
| **AmpPhase** (E14) | 29.92 / 30.39 partial | FFT instability; redundant |
| **RadialPSF unsharp gate** (E15) | 30.71 / 31.41 @8k | weaker than trunk already |
| **Large model 10M + fp16 + 2e-4** | NaN collapse | use bf16 + lr≤1e-4 + accum |
| **batch=1 same steps as batch=8** | under-trained | scale steps with effective batch |

## Neutral / noise (≤0.03 dB at 6k)

| Idea | Result |
|------|--------|
| Patch-wise amp loss | +0.01 |
| Conf-weighted HF | +0.03 |
| EMA at short schedule | early lag, catch-up at end |
| LP-KPN on top of Align (E12) | 31.75 / 32.41 ≈ E11 |

## Capacity

- Sweet spot on 8GB: **4–12M**; below ~2M underfit; ≥15M diminishing on 406 pairs.
- Paper RealSR ×2 often 33.5–34.5 Y with **100k–1M** iters — gap is budget, not missing heads.

## Sources

- `experiments/improve/MATH_HEADS_REPORT.md`
- `experiments/ablation_x2/ABLATION_REPORT.md`
- `research/realsr-optical/REPORT.md`
