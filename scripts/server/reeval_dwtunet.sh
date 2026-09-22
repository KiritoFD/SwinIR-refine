#!/usr/bin/env bash
# Salvage round-2: the dwt-unet ckpts trained fine but eval_official crashed on a
# reconstruct-args bug (now fixed).  Wait for the GPU to free (round-3 last arm),
# then re-run the OFFICIAL eval on both wavelet-UNet ckpts.  No retraining.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
L="$ROOT/experiments/diffusion/logs"
WU="$ROOT/experiments/diffusion/wave_unet"
echo "[reeval $(date +%H:%M:%S)] waiting for GPU to free..."
for _ in $(seq 1 120); do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( M < 2000 )) && break
  sleep 15
done
for name in s1_b64_dwtunet s1_b64_dwtunet_dw1; do
  CK="$WU/$name/ckpt_best.pt"; [ -f "$CK" ] || CK="$WU/$name/ckpt_last.pt"
  echo "[reeval $(date +%H:%M:%S)] $name ckpt=$(basename $CK)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$WU/$name/eval_iqa" > "$L/wu_${name}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error|state_dict" "$L/wu_${name}.eval.log" | tail -3
done
echo "[reeval $(date +%H:%M:%S)] done"
