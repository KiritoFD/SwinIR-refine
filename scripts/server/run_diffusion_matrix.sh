#!/usr/bin/env bash
# Complete the diffusion SR line: 2x2 matrix (latent|pixel) x (flow|reg),
# then official RealSR V3 evaluation for every arm.
#
#   bash scripts/server/run_diffusion_matrix.sh            # full matrix
#   SKIP_TRAIN=1 bash scripts/server/run_diffusion_matrix.sh   # eval only
set -uo pipefail

export ROOT="${ROOT:-/home/ds/realsr}"
export PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
export DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/ds/.cache/huggingface}"

VAE="${VAE:-flux1-vae}"
OUT="$ROOT/experiments/diffusion"
STEPS_LATENT="${STEPS_LATENT:-20000}"
STEPS_PIXEL="${STEPS_PIXEL:-20000}"
LATENT_BATCH="${LATENT_BATCH:-32}"
PIXEL_BATCH="${PIXEL_BATCH:-8}"
LATENT_PATCH="${LATENT_PATCH:-256}"   # LR crop -> HR 512 -> latent 64x64 (1024 tokens)
PIXEL_PATCH="${PIXEL_PATCH:-64}"      # LR crop -> HR 128 (4096 tokens)
SWEEP="${SWEEP:-2,4,8,16,32}"
SWEEP_PAIRS="${SWEEP_PAIRS:-8}"
LATENT_EVAL_STEPS="${LATENT_EVAL_STEPS:-20}"
PIXEL_EVAL_STEPS="${PIXEL_EVAL_STEPS:-8}"
TILE_BATCH="${TILE_BATCH:-8}"

mkdir -p "$OUT" "$HF_HOME"
cd "$ROOT" || exit 1

log() { echo; echo "########## $* ##########"; }

log "smoke"
"$PY" -m diffusion.smoke_test || exit 1

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  log "[1/4] latent + flow  (VAE=$VAE, HR=$((LATENT_PATCH*2)), batch=$LATENT_BATCH)"
  "$PY" -m diffusion.train_latent \
    --data-root "$DATA" --vae "$VAE" --size S --objective flow \
    --lr-patch "$LATENT_PATCH" --batch "$LATENT_BATCH" --steps "$STEPS_LATENT" \
    --amp --out "$OUT/latent_flow" || exit 1

  log "[2/4] latent + reg  (deterministic, PSNR upper bound of the latent arm)"
  "$PY" -m diffusion.train_latent \
    --data-root "$DATA" --vae "$VAE" --size S --objective reg \
    --lr-patch "$LATENT_PATCH" --batch "$LATENT_BATCH" --steps "$STEPS_LATENT" \
    --amp --out "$OUT/latent_reg" || exit 1

  log "[3/4] pixel + flow  (HR=$((PIXEL_PATCH*2)), batch=$PIXEL_BATCH)"
  "$PY" -m diffusion.train_pixel \
    --data-root "$DATA" --size S --objective flow --residual 1 \
    --lr-patch "$PIXEL_PATCH" --batch "$PIXEL_BATCH" --steps "$STEPS_PIXEL" \
    --amp --out "$OUT/pixel_flow" || exit 1

  log "[4/4] pixel + reg  (DiT regressor, comparable to SwinIR E11)"
  "$PY" -m diffusion.train_pixel \
    --data-root "$DATA" --size S --objective reg \
    --lr-patch "$PIXEL_PATCH" --batch "$PIXEL_BATCH" --steps "$STEPS_PIXEL" \
    --amp --out "$OUT/pixel_reg" || exit 1
fi

log "eval: latent flow (sweep $SWEEP on $SWEEP_PAIRS pairs, then full @ $LATENT_EVAL_STEPS)"
[[ -f "$OUT/latent_flow/ckpt_best.pt" ]] && "$PY" -m diffusion.eval_official \
  --ckpt "$OUT/latent_flow/ckpt_best.pt" --vae "$VAE" \
  --tile "$LATENT_PATCH" --pad 16 --tile-batch "$TILE_BATCH" \
  --steps "$LATENT_EVAL_STEPS" --sweep-steps "$SWEEP" --sweep-pairs "$SWEEP_PAIRS" \
  --out "$OUT/latent_flow/eval_official"

log "eval: latent reg"
[[ -f "$OUT/latent_reg/ckpt_best.pt" ]] && "$PY" -m diffusion.eval_official \
  --ckpt "$OUT/latent_reg/ckpt_best.pt" --vae "$VAE" \
  --tile "$LATENT_PATCH" --pad 16 --tile-batch "$TILE_BATCH" \
  --out "$OUT/latent_reg/eval_official"

log "eval: pixel flow (sweep, then full @ $PIXEL_EVAL_STEPS)"
[[ -f "$OUT/pixel_flow/ckpt_best.pt" ]] && "$PY" -m diffusion.eval_official \
  --ckpt "$OUT/pixel_flow/ckpt_best.pt" --mode pixel --residual \
  --tile "$PIXEL_PATCH" --pad 16 --tile-batch "$TILE_BATCH" \
  --steps "$PIXEL_EVAL_STEPS" --sweep-steps "$SWEEP" --sweep-pairs "$SWEEP_PAIRS" \
  --out "$OUT/pixel_flow/eval_official"

log "eval: pixel reg"
[[ -f "$OUT/pixel_reg/ckpt_best.pt" ]] && "$PY" -m diffusion.eval_official \
  --ckpt "$OUT/pixel_reg/ckpt_best.pt" --mode pixel \
  --tile "$PIXEL_PATCH" --pad 16 --tile-batch "$TILE_BATCH" \
  --out "$OUT/pixel_reg/eval_official"

log "SUMMARY"
"$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"{'arm':14s} {'objective':10s} {'n':>4s} {'Y-PSNR':>8s} {'Y-SSIM':>8s} {'RGB':>7s}")
for d in sorted(root.iterdir()):
    f = d / "eval_official" / "eval.json"
    if not f.is_file():
        continue
    s = json.loads(f.read_text())
    print(f"{d.name:14s} {s.get('objective','?'):10s} {s['n']:4d} "
          f"{s['psnr_y']:8.3f} {s['ssim_y']:8.4f} {s['psnr_rgb']:7.2f}")
    for r in s.get("sweep", []):
        print(f"    sweep nfe={r['nfe']:3d} n={r['n']:3d} Y {r['psnr_y']:.3f}/{r['ssim_y']:.4f}")
print("\nreference: E11 regression (ModSwinIR + Align-L1, 12k) Y 33.47 / 0.9144")
PYEOF

log "ALL DONE"
