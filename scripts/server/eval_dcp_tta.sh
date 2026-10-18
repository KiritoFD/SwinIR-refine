#!/bin/bash
set -e
source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate harness-qwen
cd /home/ds/realsr

for eta in 0.10 0.12; do
    echo "==================== DCP + TTA eta=${eta} ===================="
    tag="eval_dcp_tta_eta$(echo $eta | tr -d .)"
    python -m diffusion.eval_official \
        --ckpt experiments/diffusion/mech_best/p_shift4/ckpt_best.pt \
        --mode pixel --objective reg --tile 64 --pad 16 \
        --dcp 1 --dcp-eta ${eta} --tta --iqa \
        --out experiments/diffusion/mech_best/p_shift4/${tag}
done
