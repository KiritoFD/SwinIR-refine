# SwinIR-refine · RealSR V3 ×2

Real-camera image super-resolution on **[RealSR V3](https://github.com/csjcai/RealSR)** (Cai et al., ICCV 2019).
Minimal training/eval package for residual Mod-SwinIR + offset-aligned L1.

**Remote:** `https://github.com/KiritoFD/SwinIR-refine`

## Layout

```
model/          # production code (train / eval / metrics)
exp/            # experiment notes, verified results, negative results
data/           # RealSR(V3) local data (gitignored)
experiments/    # run artifacts (gitignored)
```

Upstream SwinIR demo code (testsets / figs / predict / main_test) was removed from this fork.
Original README: [`exp/README.upstream.md`](exp/README.upstream.md).

## Requirements

- Python 3.10+, PyTorch 2.x, CUDA GPU (tested on RTX 4070 8GB)
- RealSR V3 under `data/RealSR(V3)/{Canon,Nikon}/{Train,Test}/{2,3,4}/`
  with pairs named `*_LR2.png` / `*_HR.png`

## Train

```powershell
python -m model.train `
  --data-root "G:\RealSR\data\RealSR(V3)" `
  --out experiments\improve\E11_align_loss `
  --scale 2 --model-size base `
  --batch-size 4 --lr-patch 64 --steps 12000 `
  --lr 2e-4 --warmup 150 --num-workers 0 `
  --eval-every 2000 --eval-pairs 6 --save-every 4000 `
  --cameras Canon,Nikon --amp --align-loss --ema 0.999
```

| Flag | Meaning |
|------|---------|
| `--align-loss` | Offset-aligned L1 (default on; `--no-align-loss` to disable) |
| `--ema 0.999` | EMA weights |
| `--optimizer adamw\|muon` | Optimizer (Muon ported from sr-scaling) |
| `--model-size base` | 4.05M (current slim package) |

## Evaluate (official RealSR protocol)

Paper protocol: **train RGB, test Y**. Metrics follow official `Test.m`:
MATLAB limited-range `rgb2ycbcr` Y → uint8 PSNR/SSIM, `modcrop 4`, **no shave**, full Canon+Nikon Test.

```powershell
python -m model.eval `
  --ckpt experiments\improve\E11_align_loss\ckpt_best.pt `
  --data-root "G:\RealSR\data\RealSR(V3)" `
  --scale 2 --max-pairs 0
```

`--max-pairs 0` = full official Test (100 pairs at ×2). Use a small `N` for smoke checks.

Details: [`exp/EVAL_PROTOCOL.md`](exp/EVAL_PROTOCOL.md).

## Current progress

| Item | Status |
|------|--------|
| Best recipe | Residual bicubic + Align-L1 + EMA + 12k steps (E11, 4.05M) |
| **Official full Test (100 pairs)** | **Y 33.47 / SSIM 0.9144** · RGB 31.62 |
| A0 pure L1 same trunk | Y 33.37 → Align **+0.10 dB** |
| SwinIR-largeish L1 baseline | Y 32.97 → E11 **+0.50 dB** |
| Failed (do not re-enable) | UWCL/CX, Wiener/AmpPhase/RadialPSF, LP-KPN on Align |
| Muon A/B | training (`E11c_muon`) |

Full write-up: [`exp/OFFICIAL_EVAL.md`](exp/OFFICIAL_EVAL.md), [`exp/VERIFIED_SUMMARY.md`](exp/VERIFIED_SUMMARY.md).

## Citations

```bibtex
@inproceedings{cai2019toward,
  title={Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model},
  author={Cai, Jianrui and Zeng, Hui and Yong, Hongwei and Cao, Zisheng and Zhang, Lei},
  booktitle={ICCV},
  year={2019}
}
```

Upstream SwinIR: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR).
