#!/usr/bin/env bash
# Launch full diffusion pipeline on dserver (uses harness-qwen conda env).
set -euo pipefail
export ROOT=/home/ds/realsr
export PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
export DATA="/home/ds/realsr/data/RealSR(V3)"
VAE="${1:-flux1-dev}"
cd "$ROOT"

# prefer local HF cache; allow download
export HF_HOME="${HF_HOME:-/home/ds/.cache/huggingface}"
mkdir -p "$HF_HOME" "$ROOT/experiments/diffusion"

echo "== smoke =="
"$PY" -m diffusion.smoke_test

echo "== VAE noise floor: $VAE =="
"$PY" -m diffusion.vae_noise_floor \
  --data-root "$DATA" \
  --vae "$VAE" \
  --max-pairs "${FLOOR_PAIRS:-20}" \
  --tile 256 \
  --out "$ROOT/experiments/diffusion/vae_noise"

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "done (train skipped)"
  exit 0
fi

STEPS="${STEPS:-20000}"

echo "== latent DiT =="
"$PY" -m diffusion.train_latent \
  --data-root "$DATA" --vae "$VAE" --size S \
  --lr-patch 64 --batch "${LATENT_BATCH:-32}" \
  --steps "$STEPS" --amp \
  --out "$ROOT/experiments/diffusion/latent_dit_${VAE}"

echo "== pixel DiT HR128 =="
"$PY" -m diffusion.train_pixel \
  --data-root "$DATA" --size S \
  --lr-patch 64 --batch "${PIXEL_BATCH:-8}" \
  --steps "$STEPS" --amp --grad-ckpt --residual \
  --out "$ROOT/experiments/diffusion/pixel_dit_hr128"

echo "== official eval latent =="
"$PY" -m diffusion.eval_official \
  --ckpt "$ROOT/experiments/diffusion/latent_dit_${VAE}/ckpt_best.pt" \
  --vae "$VAE" --mode latent \
  --data-root "$DATA" --max-pairs "${EVAL_PAIRS:-0}" \
  --steps "${EVAL_STEPS:-20}" --tile 128 \
  --out "$ROOT/experiments/diffusion/latent_dit_${VAE}/eval_official"

echo "== official eval pixel =="
"$PY" -m diffusion.eval_official \
  --ckpt "$ROOT/experiments/diffusion/pixel_dit_hr128/ckpt_best.pt" \
  --mode pixel --residual \
  --data-root "$DATA" --max-pairs "${EVAL_PAIRS:-0}" \
  --steps "${EVAL_STEPS:-20}" --tile 128 \
  --out "$ROOT/experiments/diffusion/pixel_dit_hr128/eval_official"

echo "== ALL DONE =="
