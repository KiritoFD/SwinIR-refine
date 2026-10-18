#!/usr/bin/env bash
# Learnability gate for the new baseline backbones — runs on the 4090 only.
# Nothing is launched on the local box; the gate must pass before any GPU hours
# are spent on a real arm (SRResNet silently collapsed last time: VAL peaked at
# 5k steps then fell to 20.9 dB, and no pre-flight caught it).
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python
find diffusion scripts/server -name '*.py' -o -name '*.sh' | xargs -r sed -i 's/\r$//'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
$P -u -m diffusion.smoke_bl_archs --steps 2000 --bs 4 --hr 128 --lr 3e-4
echo "gate_exit=$?"
