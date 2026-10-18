#!/bin/bash
set -e
source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate harness-qwen
cd /home/ds/realsr

OUT="experiments/diffusion/wavesr_physics_b64"
mkdir -p "$OUT"

echo "==================== 1. TRAIN WAVESR-PHYSICS (Base 64, Batch 240, VRAM ~38GB) ===================="
echo "Priors enabled: Optical-Conditioned ResUNet + LL-Swap + Bounded Sub-Pixel Align + CA Loss + Muon"

python -m diffusion.train_pixel \
  --data-root "data/RealSR(V3)" \
  --out "$OUT" \
  --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --batch 240 --amp \
  --grad-ckpt \
  --num-workers 8 --cache-data 0 --warmup 500 --steps 12000 --ema 0.999 \
  --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 \
  --optical-cond \
  --ll-swap \
  --align-loss --align-max-shift 0.5 \
  --ca-loss --ca-weight 0.005 \
  --ycbcr-loss --ycbcr-y-weight 1.0 --ycbcr-chroma-weight 0.2 --chroma-tv-weight 0.005 \
  --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 4000 --val-eval-steps 8 \
  --save-every 2000 \
  --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5 2>&1 | tee "$OUT/train.log"

echo "==================== 2. OFFICIAL EVAL (100 PAIRS, Tile-Batch 128, Full 8 Metrics) ===================="
python -m diffusion.eval_official \
  --ckpt "$OUT/ckpt_best.pt" \
  --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 128 \
  --iqa \
  --out "$OUT/eval_raw" 2>&1 | tee "$OUT/eval_raw.log"

echo "==================== 3. OFFICIAL EVAL + ADAPTIVE CA-WD-DCP (100 PAIRS, Tile-Batch 128) ===================="
python -m diffusion.eval_official \
  --ckpt "$OUT/ckpt_best.pt" \
  --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 128 \
  --dcp 1 --dcp-eta 0.12 --dcp-decay 0.0 \
  --dcp-sigma-canon "0.65,0.60,0.55" --dcp-sigma-nikon "0.50,0.50,0.45" \
  --iqa \
  --out "$OUT/eval_wddcp_adaptive" 2>&1 | tee "$OUT/eval_wddcp_adaptive.log"

echo "==================== 4. OFFICIAL EVAL + ADAPTIVE CA-WD-DCP + TTA (100 PAIRS, Tile-Batch 128) ===================="
python -m diffusion.eval_official \
  --ckpt "$OUT/ckpt_best.pt" \
  --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 128 \
  --dcp 1 --dcp-eta 0.12 --dcp-decay 0.0 \
  --dcp-sigma-canon "0.65,0.60,0.55" --dcp-sigma-nikon "0.50,0.50,0.45" \
  --tta --iqa \
  --out "$OUT/eval_wddcp_adaptive_tta" 2>&1 | tee "$OUT/eval_wddcp_adaptive_tta.log"
