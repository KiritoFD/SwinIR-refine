# Results — what works on RealSR V3 ×2

**Date:** 2026-09-14 · **GPU:** RTX 4070 8GB · **Data:** Canon+Nikon, 406 train pairs ×2

## Recipe (keep)

| Piece | Role |
|-------|------|
| ModSwinIR v2 trunk | OCA + GDFN + AMF(tanh) + FiLM-RAPE + DCN |
| Residual recon | `ŷ = bicubic(x) + f_θ(x)`, zero-init head |
| **Offset-aligned L1** | train-only warp HR→SR (±3 px), α=0.25 pure L1 |
| L1 primary + EMA 0.999 | PSNR track |
| Y-channel eval | official RealSR style (RGB-train / Y-eval) |

## Numbers (12-image tiled Test, RGB / Y)

| Model | RGB | Y |
|-------|-----|---|
| SwinIR-light 0.61M L1 | 31.07 | — |
| SwinIR-largeish 3.96M L1 | 31.16 | — |
| Mod v2 L1+EMA | 31.09 | — |
| **Mod v2 + Align (E11, 12k)** | **31.73** | **32.40** |
| Full 100-pair Test (E9-style, RGB) | ~31.7 | — |

**Best production ckpt:** `experiments/improve/E11_align_loss/ckpt_best.pt`

## Commands

```powershell
# train
python -m model.train --data-root "G:\RealSR\data\RealSR(V3)" --out experiments\E11 \
  --scale 2 --arch mod --model-size base --batch-size 8 --lr-patch 64 --steps 12000 \
  --l1-only --align-loss --ema 0.999 --amp --eval-every 1500 --save-every 3000

# eval
python -m model.eval --ckpt experiments\E11\ckpt_best.pt --scale 2 --max-pairs 12
```
