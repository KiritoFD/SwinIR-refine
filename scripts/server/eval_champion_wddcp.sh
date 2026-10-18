#!/bin/bash
set -e
source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate harness-qwen
cd /home/ds/realsr

CKPT="experiments/diffusion/mech_best/p_shift4/ckpt_best.pt"

echo "==================== 1. CHAMPION RAW (FULL 8 METRICS) ===================="
python -m diffusion.eval_official \
    --ckpt "$CKPT" \
    --mode pixel --objective reg --tile 64 --pad 16 \
    --iqa \
    --out experiments/diffusion/mech_best/p_shift4/eval_full_metrics_raw

echo "==================== 2. CHAMPION + WD-DCP (OPTICAL PSF sigma=0.6, hf_decay=0.0, eta=0.12) ===================="
python -m diffusion.eval_official \
    --ckpt "$CKPT" \
    --mode pixel --objective reg --tile 64 --pad 16 \
    --dcp 1 --dcp-eta 0.12 --dcp-decay 0.0 --dcp-sigma 0.6 \
    --iqa \
    --out experiments/diffusion/mech_best/p_shift4/eval_full_metrics_wddcp

echo "==================== 3. CHAMPION + WD-DCP + TTA (FULL 8 METRICS) ===================="
python -m diffusion.eval_official \
    --ckpt "$CKPT" \
    --mode pixel --objective reg --tile 64 --pad 16 \
    --dcp 1 --dcp-eta 0.12 --dcp-decay 0.0 --dcp-sigma 0.6 \
    --tta --iqa \
    --out experiments/diffusion/mech_best/p_shift4/eval_full_metrics_wddcp_tta
