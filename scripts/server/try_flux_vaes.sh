#!/usr/bin/env bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
# re-upload vae.py first from parent
for v in flux1-vae flux2-vae sdxl-vae; do
  echo "===== TRY $v ====="
  $PY scripts/server/download_vae.py --vae "$v" 2>&1 | tail -20
  echo
done
# data size
ls -lah /home/ds/realsr/data/
