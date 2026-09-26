# Documentation map

One-page index of every document in this repo. Start at the root `README.md` (the
paper-style report); everything below it is deeper detail.

## Reports (read in this order)
| # | file | what it is |
|---|---|---|
| 1 | [`../README.md`](../README.md) | **Paper-style report** — headline results & baseline comparison tables, method, upper bound, reproduce, citations. |
| 2 | [`../docs/baseline/README.md`](baseline/README.md) | **Baseline catalogue (2021–2026)** — every representative method, its numbers (source + protocol tag), and which we reproduced on the server. |
| 3 | [`../exp/RELATED_WORK_AND_UPPER_BOUND.md`](../exp/RELATED_WORK_AND_UPPER_BOUND.md) | **Related work + the empirical PSNR/SSIM ceiling** — verified citations + our training-free three-wall measurement. |
| 4 | [`../exp/FINAL_REPORT.md`](../exp/FINAL_REPORT.md) | **Full experiment chronicle** (2026-09-10 → 09-26): every run's hypothesis/config/result/verdict, appendices A–F (106-run master table, root inventory, per-run config ledger, audit findings, upper-bound probe). |
| 5 | [`slides_final.pptx`](slides_final.pptx) | **中文学术报告幻灯片（22 页）**：引言→相关工作→方法(含数学)→实验(含全方法全指标密集大表)。生成器 [`make_slides.py`](make_slides.py) + 配图 [`make_figs.py`](make_figs.py)。 |
| 6 | [`slides.md`](slides.md) | 英文版 Marp 大纲（可 `marp` 导出）；内容与 5 对应。 |

## Tooling & reproduction
| file | purpose |
|---|---|
| `../diffusion/oracle.py` | training-free upper-bound probes (E_align / E_null / E_noise(MAD) / composite) — `python -m diffusion.oracle --scale 2\|3\|4` |
| `../diffusion/baselines_sr.py` | classic SR baselines (EDSR / RCAN / SRResNet / RRDB) — `--backbone edsr\|rcan\|srresnet\|rrdb` |
| `../scripts/server/run_baselines.sh`, `run_baselines2.sh` | reproduce the baselines on RealSR Train (A-strict protocol) |
| `../scripts/server/run_*.sh` | every experiment campaign (tmux + per-arm GPU wait + idempotent `eval.json` skip) |

## Stage notes (appendices to the chronicle)
`exp/STAGE_SUMMARY.md` · `exp/PROGRESS_REPORT.md` · `exp/RESULTS_SUMMARY.md` ·
`exp/WAVE_ARMS.md` · `exp/MAMBA_MATRIX_RESULTS.md` · `exp/NEW_ARMS.md` ·
`exp/EVAL_PROTOCOL.md` · `exp/OFFICIAL_EVAL.md` · `exp/VERIFIED_SUMMARY.md`

## Conventions
- **Metrics**: PSNR-Y is secondary; **SSIM / MUSIQ / MANIQA** decide. ±0.05 dB / small
  perceptual deltas = tie. Official `Test.m`: Limited-range Y, uint8, modcrop4, no shave, 100 pairs.
- **Protocol tags**: A-strict (our paired protocol) · B-128/512 (diffusion-paper crop) ·
  C-REdeg (Real-ESRGAN re-degraded) · D-blind (pretrained, synthetic-trained). Never compare across tags.
- **Compute rule**: one b64 stride-1 arm saturates the 4090 → serial campaigns, no grad-ckpt,
  drop batch when memory-bound.
