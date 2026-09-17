#!/usr/bin/env bash
# 36h plan v3 — arms 4..8 (the ones not yet done when torch.compile landed).
# Identical to v2 except --compile; set COMPILE=0 to disable.
#
# Arms 1-3 (latent_flow_flux, latent_reg_flux, pixel_reg) were trained by
# run_36h2.sh WITHOUT compile, so this script must not be used to compare
# wall-clock against them — only accuracy.
set -uo pipefail

export ROOT="${ROOT:-/home/ds/realsr}"
export PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
export DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/ds/.cache/huggingface}"
cd "$ROOT" || exit 1

OUT="$ROOT/experiments/diffusion"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"

C=""
if [[ "${COMPILE:-1}" != "0" ]]; then C="--compile"; fi

LAT="--lr-patch 256 --batch 64 --amp --num-workers 8 --eval-every 1000 \
     --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000 $C"
LAT_XS="--lr-patch 256 --batch 96 --amp --num-workers 8 --eval-every 1000 \
        --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000 $C"
PIX="--lr-patch 64 --batch 28 --amp --num-workers 8 --eval-every 1000 \
     --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000 $C"
PIX_XS="--lr-patch 64 --batch 32 --amp --num-workers 8 --eval-every 1000 \
        --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000 $C"

ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

T0=$(date +%s)
DEADLINE_H="${DEADLINE_H:-33}"
EVAL_DEADLINE_H="${EVAL_DEADLINE_H:-35.5}"
EVAL_DEADLINE_S=$(awk -v h="$EVAL_DEADLINE_H" 'BEGIN{printf "%d", h*3600}')
_now() { date +%s; }
elapsed_h() { awk -v a=$(( $(_now) - T0 )) 'BEGIN{printf "%.1f", a/3600}'; }
past() { (( $(_now) - T0 > $1 )); }

train_latent() {
  local tag=$1 vae=$2 size=$3 obj=$4 steps=$5; shift 5
  if past $(( DEADLINE_H * 3600 )); then
    say "SKIP TRAIN $tag — past ${DEADLINE_H}h deadline (elapsed $(elapsed_h)h)"; return 1
  fi
  say "TRAIN $tag  (vae=$vae size=$size objective=$obj steps=$steps) elapsed=$(elapsed_h)h"
  "$PY" -m diffusion.train_latent --data-root "$DATA" --out "$OUT/$tag" \
    --vae "$vae" --latent-cache "data/latents/$vae" \
    --size "$size" --objective "$obj" --steps "$steps" "$@" \
    2>&1 | tee "$LOG/$tag.train.log" | grep -E "LatentDiT|compile|VAL |EARLY|done|Error|Traceback" | tail -40
}

train_pixel() {
  local tag=$1 size=$2 obj=$3 steps=$4; shift 4
  if past $(( DEADLINE_H * 3600 )); then
    say "SKIP TRAIN $tag — past ${DEADLINE_H}h deadline (elapsed $(elapsed_h)h)"; return 1
  fi
  say "TRAIN $tag  (pixel size=$size objective=$obj steps=$steps) elapsed=$(elapsed_h)h"
  "$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT/$tag" \
    --size "$size" --objective "$obj" --steps "$steps" "$@" \
    2>&1 | tee "$LOG/$tag.train.log" | grep -E "PixelDiT|compile|VAL |EARLY|done|Error|Traceback" | tail -40
}

eval_latent() {
  local tag=$1 vae=$2
  [[ -f "$OUT/$tag/ckpt_best.pt" ]] || { echo "  no ckpt for $tag"; return; }
  if past "$EVAL_DEADLINE_S"; then
    say "SKIP EVAL $tag — past ${EVAL_DEADLINE_H}h (elapsed $(elapsed_h)h)"; return 1
  fi
  say "EVAL $tag  elapsed=$(elapsed_h)h"
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/$tag/ckpt_best.pt" --vae "$vae" --mode latent \
    --tile 256 --pad 16 --tile-batch 8 --steps 20 \
    --sweep-steps 2,4,8,16,32 --sweep-pairs 8 \
    --out "$OUT/$tag/eval_official" 2>&1 | tee "$LOG/$tag.eval.log" | grep -E "OFFICIAL|SWEEP|FAIL" | tail -10
}

eval_pixel() {
  local tag=$1; shift
  [[ -f "$OUT/$tag/ckpt_best.pt" ]] || { echo "  no ckpt for $tag"; return; }
  if past "$EVAL_DEADLINE_S"; then
    say "SKIP EVAL $tag — past ${EVAL_DEADLINE_H}h (elapsed $(elapsed_h)h)"; return 1
  fi
  say "EVAL $tag  elapsed=$(elapsed_h)h"
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/$tag/ckpt_best.pt" --mode pixel "$@" \
    --tile 64 --pad 16 --tile-batch 8 --steps 8 \
    --sweep-steps 2,4,8,16,32 --sweep-pairs 8 \
    --out "$OUT/$tag/eval_official" 2>&1 | tee "$LOG/$tag.eval.log" | grep -E "OFFICIAL|SWEEP|FAIL" | tail -10
}

say "SETUP v3  compile=${C:-off}"
"$PY" -m diffusion.smoke_test 2>&1 | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ------------------------------------------------- 4. pixel / flow
train_pixel  pixel_flow S flow 16000 $PIX
eval_pixel   pixel_flow --objective flow --residual

# ------------------------------------------------- 5/6. param-matched XS
train_latent latent_flow_flux_XS flux1-vae XS flow 22000 $LAT_XS
eval_latent  latent_flow_flux_XS flux1-vae

train_pixel  pixel_reg_XS XS reg 14000 $PIX_XS
eval_pixel   pixel_reg_XS --objective reg

# ------------------------------------------------- 7/8. VAE ablation (4ch)
train_latent latent_flow_sd sd-vae-ft-ema S flow 16000 $LAT
eval_latent  latent_flow_sd sd-vae-ft-ema

train_latent latent_flow_sdxl sdxl-vae S flow 13000 $LAT
eval_latent  latent_flow_sdxl sdxl-vae

say "SUMMARY"
"$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ref = 33.47
print(f"{'arm':22s} {'obj':5s} {'n':>4s} {'Y-PSNR':>8s} {'Y-SSIM':>8s} {'vs E11':>8s}")
for d in sorted(root.iterdir()):
    f = d / "eval_official" / "eval.json"
    if not d.is_dir() or not f.is_file():
        continue
    s = json.loads(f.read_text())
    print(f"{d.name:22s} {s.get('objective','?'):5s} {s['n']:4d} "
          f"{s['psnr_y']:8.3f} {s['ssim_y']:8.4f} {s['psnr_y']-ref:+8.3f}")
    for r in s.get("sweep", []):
        print(f"      sweep nfe={r['nfe']:3d} n={r['n']:3d} Y {r['psnr_y']:.3f}")
print(f"\nreference: E11 ModSwinIR+Align (4.05M, 12k) Y {ref} / 0.9144")
PYEOF

say "ALL DONE [$(ts)]"
