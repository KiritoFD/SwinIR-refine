#!/usr/bin/env bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
for v in sd-vae-ft-ema sd-vae-ft-mse sd3-vae flux1-schnell kl-f4; do
  echo "===== TRY $v ====="
  $PY scripts/server/download_vae.py --vae "$v" 2>&1 | tail -15
  echo
done
