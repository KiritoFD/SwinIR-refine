# ModSwinIR (kept)

RealSR V3 ×2 residual transformer + **offset-aligned L1**.

```
model/
  layers.py    # OCA, GDFN, AMF-v2 (tanh), FiLM-RAPE, DeformAlign
  model.py     # residual bicubic trunk (4.05M base)
  dataset.py   # PIL-crop patches (low RAM)
  losses.py    # OffsetAlignedLoss (train-only alignment)
  train.py     # L1 / Align + EMA
  eval.py      # tiled PSNR/SSIM + Y-channel
```

Train: `python -m model.train --data-root ... --align-loss --ema 0.999`  
Eval:  `python -m model.eval --ckpt .../ckpt_best.pt --max-pairs 12`

See `../exp/RESULTS.md` and `../exp/NEGATIVE.md`.
