#!/bin/bash
set -e
source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate harness-qwen
cd /home/ds/realsr

for eta in 0.05 0.15 0.20; do
    echo "==================== DCP eta=${eta} ===================="
    tag="eval_dcp_eta$(echo $eta | tr -d .)"
    python -m diffusion.eval_official \
        --ckpt experiments/diffusion/mech_best/p_shift4/ckpt_best.pt \
        --mode pixel --objective reg --tile 64 --pad 16 \
        --dcp 1 --dcp-eta ${eta} \
        --out experiments/diffusion/mech_best/p_shift4/${tag}
done
