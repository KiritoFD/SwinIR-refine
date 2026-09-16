#!/usr/bin/env bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
OUT=/home/ds/realsr/experiments/diffusion/vae_noise
mkdir -p "$OUT"
for v in sdxl-vae sd-vae-ft-ema sd-vae-ft-mse; do
  echo "===== $v ====="
  $PY -m diffusion.vae_noise_floor \
    --data-root '/home/ds/realsr/data/RealSR(V3)' \
    --vae "$v" --max-pairs 15 --tile 192 --out "$OUT" \
    > "$OUT/run_${v}.log" 2>&1
  tail -5 "$OUT/run_${v}.log"
done
echo DONE_COMPARE
ls -la "$OUT"
