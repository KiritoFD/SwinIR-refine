#!/usr/bin/env bash
# Complete server pipeline: VAE floor → latent DiT → pixel DiT → official eval.
set -euo pipefail

VAE="${1:-flux1-dev}"
DATA="${2:-$HOME/realsr/data/RealSR(V3)}"
ROOT="${ROOT:-$HOME/realsr}"
PY="${PY:-python3}"
OUT="$ROOT/experiments/diffusion"
STEPS="${STEPS:-20000}"
EVAL_STEPS="${EVAL_STEPS:-20}"
EVAL_PAIRS="${EVAL_PAIRS:-0}"   # 0 = full Test

echo "== host $(hostname) =="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cd "$ROOT"
mkdir -p "$OUT"

echo "== [0] smoke =="
"$PY" -m diffusion.smoke_test

echo "== [1] VAE noise floor $VAE =="
"$PY" -m diffusion.vae_noise_floor \
  --data-root "$DATA" --vae "$VAE" --max-pairs "${FLOOR_PAIRS:-20}" \
  --tile 256 --out "$OUT/vae_noise"

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "SKIP_TRAIN=1 → stop after VAE floor"
  exit 0
fi

echo "== [2] latent DiT =="
"$PY" -m diffusion.train_latent \
  --data-root "$DATA" --vae "$VAE" --size S \
  --lr-patch 64 --batch "${LATENT_BATCH:-16}" \
  --steps "$STEPS" --amp \
  --out "$OUT/latent_dit_${VAE}"

echo "== [3] pixel DiT HR128 =="
"$PY" -m diffusion.train_pixel \
  --data-root "$DATA" --size S \
  --lr-patch 64 --batch "${PIXEL_BATCH:-4}" \
  --steps "$STEPS" --amp --grad-ckpt --residual \
  --out "$OUT/pixel_dit_hr128"

echo "== [4] official eval =="
"$PY" -m diffusion.eval_official \
  --ckpt "$OUT/latent_dit_${VAE}/ckpt_best.pt" \
  --vae "$VAE" --mode latent \
  --data-root "$DATA" --max-pairs "$EVAL_PAIRS" \
  --steps "$EVAL_STEPS" --tile 128 \
  --out "$OUT/latent_dit_${VAE}/eval_official"

"$PY" -m diffusion.eval_official \
  --ckpt "$OUT/pixel_dit_hr128/ckpt_best.pt" \
  --mode pixel --residual \
  --data-root "$DATA" --max-pairs "$EVAL_PAIRS" \
  --steps "$EVAL_STEPS" --tile 128 \
  --out "$OUT/pixel_dit_hr128/eval_official"

echo "== ALL DONE =="
echo "VAE:    $OUT/vae_noise"
echo "Latent: $OUT/latent_dit_${VAE}/eval_official/eval.json"
echo "Pixel:  $OUT/pixel_dit_hr128/eval_official/eval.json"
