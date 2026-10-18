#!/bin/bash
set -e
source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate harness-qwen
cd /home/ds/realsr

OUT="experiments/diffusion/test_vram36g"
mkdir -p "$OUT"

echo "==================== Testing VRAM Saturation (>=36GB) ===================="
python -m diffusion.train_pixel \
  --data-root "data/RealSR(V3)" \
  --out "$OUT" \
  --backbone unet --size S --base 96 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --batch 240 --amp \
  --grad-ckpt \
  --num-workers 8 --cache-data 0 --warmup 50 --steps 20 --ema 0.999 --dwt-loss --dwt-weight 8 \
  --dwt-levels 2 --eval-every 50 --val-pairs 8 --min-steps 10 --save-every 50 \
  --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5
