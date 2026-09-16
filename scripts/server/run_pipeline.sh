#!/usr/bin/env bash
# Server setup + experiment launch for RealSR diffusion line.
# Usage: bash scripts/server/run_pipeline.sh [vae_preset] [data_root]
set -euo pipefail

VAE="${1:-flux1-dev}"
DATA="${2:-$HOME/realsr/data/RealSR(V3)}"
ROOT="${ROOT:-$HOME/realsr}"
PY="${PY:-python3}"
OUT="$ROOT/experiments/diffusion"

echo "== host: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$OUT"
cd "$ROOT"

echo "== [0] smoke DiT (no VAE) =="
"$PY" -m diffusion.smoke_test

echo "== [1] VAE noise floor: $VAE =="
"$PY" -m diffusion.vae_noise_floor \
  --data-root "$DATA" \
  --vae "$VAE" \
  --max-pairs 20 \
  --tile 256 \
  --out "$OUT/vae_noise"

echo "== [2] latent DiT (short smoke 200 steps if SMOKE=1) =="
STEPS="${STEPS:-20000}"
if [[ "${SMOKE:-0}" == "1" ]]; then STEPS=200; fi
"$PY" -m diffusion.train_latent \
  --data-root "$DATA" \
  --vae "$VAE" \
  --size S \
  --lr-patch 64 \
  --batch "${LATENT_BATCH:-16}" \
  --steps "$STEPS" \
  --amp \
  --out "$OUT/latent_dit_${VAE}"

echo "== [3] pixel DiT HR=128 =="
"$PY" -m diffusion.train_pixel \
  --data-root "$DATA" \
  --size S \
  --lr-patch 64 \
  --batch "${PIXEL_BATCH:-4}" \
  --steps "$STEPS" \
  --amp \
  --grad-ckpt \
  --residual \
  --out "$OUT/pixel_dit_hr128"

echo "== DONE =="
echo "VAE floor: $OUT/vae_noise"
echo "Latent:    $OUT/latent_dit_${VAE}"
echo "Pixel:     $OUT/pixel_dit_hr128"
