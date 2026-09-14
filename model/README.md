# model package

RealSR V3 ×2 residual Mod-SwinIR + offset-aligned L1.

```
model/
  layers.py    # OCA, GDFN, AMF, FiLM, DeformAlign
  model.py     # residual bicubic trunk
  dataset.py   # PIL-crop patches
  losses.py    # OffsetAlignedLoss (train-only)
  metrics.py   # official RealSR Y / PSNR / SSIM
  optim.py     # AdamW + Muon
  train.py
  eval.py
```

See repo root `README.md` and `../exp/EVAL_PROTOCOL.md`.
