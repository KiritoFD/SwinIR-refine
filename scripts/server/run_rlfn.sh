#!/bin/bash
set -e
source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate harness-qwen
cd /home/ds/realsr

OUT="experiments/diffusion/baselines/rlfn_x2"
mkdir -p "$OUT"

echo "==================== TRAIN RLFN (A-strict baseline) ===================="
python -m diffusion.train_pixel \
    --data-root "data/RealSR(V3)" \
    --out "$OUT" \
    --backbone rlfn \
    --size S \
    --objective reg \
    --residual 1 \
    --scale 2 \
    --lr-patch 64 \
    --batch 32 \
    --amp \
    --reg-loss l1 \
    --optimizer adamw \
    --lr 1e-4 \
    --warmup 500 \
    --steps 40000 \
    --ema 0.999 \
    --eval-every 2000 \
    --patience 20 \
    --min-steps 8000 \
    --seed 1234 2>&1 | tee "$OUT/train.log"

echo "==================== EVAL RLFN (A-strict pad16 + IQA) ===================="
python -m diffusion.eval_official \
    --ckpt "$OUT/ckpt_best.pt" \
    --mode pixel \
    --objective reg \
    --tile 64 \
    --pad 16 \
    --iqa \
    --out "$OUT/eval_iqa" 2>&1 | tee "$OUT/eval.log"

echo "==================== EVAL-TTA RLFN (8-fold D4) ===================="
python -m diffusion.eval_official \
    --ckpt "$OUT/ckpt_best.pt" \
    --mode pixel \
    --objective reg \
    --tile 64 \
    --pad 16 \
    --tta \
    --iqa \
    --out "$OUT/eval_iqa_tta" 2>&1 | tee "$OUT/eval_tta.log"
