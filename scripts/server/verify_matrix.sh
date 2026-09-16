#!/usr/bin/env bash
# Smoke-run every arm of the 36h matrix for a few steps.
# Confirms: batch fits in 44-48G, val/early-stop path works, ckpt saves.
#   bash scripts/server/verify_matrix.sh            # all arms, 20 steps
#   STEPS=40 bash scripts/server/verify_matrix.sh
set -uo pipefail

export ROOT="${ROOT:-/home/ds/realsr}"
export PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
export DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

STEPS="${STEPS:-20}"
OUT="$ROOT/experiments/diffusion/_verify"
mkdir -p "$OUT"
COMMON="--data-root $DATA --steps $STEPS --amp --eval-every $STEPS --val-pairs 4 \
  --min-steps 100000 --save-every 100000 --warmup 10"

run() {
  local tag="$1"; shift
  echo; echo "########## $tag ##########"
  "$PY" -m "$@" $COMMON --out "$OUT/$tag" 2>&1 | grep -vE "Warning|warn" | tail -6
  ls -la "$OUT/$tag"/ckpt_*.pt 2>/dev/null | sed 's|.*/||'
}

run latent_flow_flux   diffusion.train_latent --objective flow --vae flux1-vae \
  --latent-cache data/latents/flux1-vae --size S --lr-patch 256 --batch 32
run latent_reg_flux    diffusion.train_latent --objective reg --vae flux1-vae \
  --latent-cache data/latents/flux1-vae --size S --lr-patch 256 --batch 32
run latent_flow_sd     diffusion.train_latent --objective flow --vae sd-vae-ft-ema \
  --latent-cache data/latents/sd-vae-ft-ema --size S --lr-patch 256 --batch 32
run latent_flow_sdxl   diffusion.train_latent --objective flow --vae sdxl-vae \
  --latent-cache data/latents/sdxl-vae --size S --lr-patch 256 --batch 32
run latent_flow_flux_XS diffusion.train_latent --objective flow --vae flux1-vae \
  --latent-cache data/latents/flux1-vae --size XS --lr-patch 256 --batch 32
run pixel_flow         diffusion.train_pixel --objective flow --size S --lr-patch 64 --batch 16
run pixel_reg          diffusion.train_pixel --objective reg --size S --lr-patch 64 --batch 16
run pixel_reg_XS       diffusion.train_pixel --objective reg --size XS --lr-patch 64 --batch 16

echo; echo "########## VERIFY DONE ##########"
