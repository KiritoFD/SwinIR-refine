#!/bin/bash
set -e
source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate harness-qwen
cd /home/ds/realsr

OUT="experiments/diffusion/scale96_muon"
mkdir -p "$OUT"

echo "==================== 1. TRAIN SCALE-96 (41.19M, Batch 240, VRAM ~38.2GB) ===================="
python -m diffusion.train_pixel \
  --data-root "data/RealSR(V3)" \
  --out "$OUT" \
  --backbone unet --size S --base 96 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --batch 240 --amp \
  --grad-ckpt \
  --num-workers 8 --cache-data 0 --warmup 500 --steps 12000 --ema 0.999 \
  --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 \
  --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 4000 --val-eval-steps 8 \
  --save-every 2000 \
  --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5 2>&1 | tee "$OUT/train.log"

echo "==================== 2. OFFICIAL EVAL (100 PAIRS, Tile-Batch 128, Full 8 Metrics) ===================="
python -m diffusion.eval_official \
  --ckpt "$OUT/ckpt_best.pt" \
  --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 128 \
  --iqa \
  --out "$OUT/eval_full_metrics_raw" 2>&1 | tee "$OUT/eval_raw.log"

echo "==================== 3. OFFICIAL EVAL + WD-DCP (100 PAIRS, Tile-Batch 128) ===================="
python -m diffusion.eval_official \
  --ckpt "$OUT/ckpt_best.pt" \
  --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 128 \
  --dcp 1 --dcp-eta 0.12 --dcp-decay 0.0 --dcp-sigma 0.6 \
  --iqa \
  --out "$OUT/eval_full_metrics_wddcp" 2>&1 | tee "$OUT/eval_wddcp.log"

echo "==================== 4. OFFICIAL EVAL + WD-DCP + TTA (100 PAIRS, Tile-Batch 128) ===================="
python -m diffusion.eval_official \
  --ckpt "$OUT/ckpt_best.pt" \
  --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 128 \
  --dcp 1 --dcp-eta 0.12 --dcp-decay 0.0 --dcp-sigma 0.6 \
  --tta --iqa \
  --out "$OUT/eval_full_metrics_wddcp_tta" 2>&1 | tee "$OUT/eval_wddcp_tta.log"

echo "==================== 5. OFFICIAL EVAL + ADAPTIVE CA-WD-DCP + TTA (100 PAIRS, Tile-Batch 128) ===================="
python -m diffusion.eval_official \
  --ckpt "$OUT/ckpt_best.pt" \
  --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 128 \
  --dcp 1 --dcp-eta 0.12 --dcp-decay 0.0 \
  --dcp-sigma-canon "0.65,0.60,0.55" --dcp-sigma-nikon "0.50,0.50,0.45" \
  --tta --iqa \
  --out "$OUT/eval_full_metrics_adaptive_ca_wddcp_tta" 2>&1 | tee "$OUT/eval_adaptive_ca_wddcp_tta.log"

echo "==================== 6. AUTOMATIC CHAIN: LAUNCH WAVESR-PHYSICS ARM ===================="
bash /home/ds/realsr/run_physics_arms.sh

