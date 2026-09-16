#!/usr/bin/env bash
# =====================================================================
# 36h serial plan v2 — RealSR V3 x2 diffusion matrix (8 arms, 1x RTX 4090 48G)
#
# v2 changes (all driven by measurement, see experiments/diffusion/bench_batch*.log):
#   * batch sizes are the MEASURED throughput-optimal ones, not "fill the VRAM":
#       latent-S  64  (23.8G, 132.0 smp/s)  b=112 -> 41.0G but 125.1 smp/s  (slower)
#       latent-XS 96  (12.0G, 327.7 smp/s)  b=320 -> 39.5G but 209.7 smp/s  (much slower)
#       pixel-S   28  (40.8G,  29.6 smp/s)  b=24  -> 35.1G,    28.0 smp/s   (slightly slower)
#       pixel-XS  32  (15.7G,  62.7 smp/s)  b=64  -> 31.2G,    56.0 smp/s   (slower)
#     Larger batches raise VRAM occupancy but LOWER samples/sec on this card:
#     above ~40G the 4090 goes super-linear (bandwidth / allocator pressure).
#   * step counts re-derived from measured s/step. v1 used optimistic pixel
#     estimates and would have needed ~35h of training before any eval.
#   * arm order: the two param-matched XS arms now run BEFORE the VAE ablation,
#     so if the clock runs out we lose the least important arms.
#   * num-workers raised (box has 24 cores; pixel arms were partly worker-bound).
#
# Budget: ~24h train + ~6h eval ~= 30h worst case; early stopping should land ~22h.
# =====================================================================
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

LAT="--lr-patch 256 --batch 64 --amp --num-workers 8 --eval-every 1000 \
     --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000"
LAT_XS="--lr-patch 256 --batch 96 --amp --num-workers 8 --eval-every 1000 \
        --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000"
PIX="--lr-patch 64 --batch 28 --amp --num-workers 8 --eval-every 1000 \
     --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000"
PIX_XS="--lr-patch 64 --batch 32 --amp --num-workers 8 --eval-every 1000 \
        --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000"

# ---------------------------------------------------------------- helpers
ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

T0=$(date +%s)
DEADLINE_H="${DEADLINE_H:-33}"        # no NEW training arm starts after this
EVAL_DEADLINE_H="${EVAL_DEADLINE_H:-35.5}"
EVAL_DEADLINE_S=$(awk -v h="$EVAL_DEADLINE_H" 'BEGIN{printf "%d", h*3600}')
_now() { date +%s; }
elapsed_h() { awk -v a=$(( $(_now) - T0 )) 'BEGIN{printf "%.1f", a/3600}'; }
past() { (( $(_now) - T0 > $1 )); }

train_latent() {           # tag vae size objective steps extra...
  local tag=$1 vae=$2 size=$3 obj=$4 steps=$5; shift 5
  if past $(( DEADLINE_H * 3600 )); then
    say "SKIP TRAIN $tag — past ${DEADLINE_H}h deadline (elapsed $(elapsed_h)h)"
    return 1
  fi
  say "TRAIN $tag  (vae=$vae size=$size objective=$obj steps=$steps) elapsed=$(elapsed_h)h"
  "$PY" -m diffusion.train_latent --data-root "$DATA" --out "$OUT/$tag" \
    --vae "$vae" --latent-cache "data/latents/$vae" \
    --size "$size" --objective "$obj" --steps "$steps" "$@" \
    2>&1 | tee "$LOG/$tag.train.log" | grep -E "LatentDiT|VAL |EARLY|done|Error|Traceback" | tail -40
}

train_pixel() {
  local tag=$1 size=$2 obj=$3 steps=$4; shift 4
  if past $(( DEADLINE_H * 3600 )); then
    say "SKIP TRAIN $tag — past ${DEADLINE_H}h deadline (elapsed $(elapsed_h)h)"
    return 1
  fi
  say "TRAIN $tag  (pixel size=$size objective=$obj steps=$steps) elapsed=$(elapsed_h)h"
  "$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT/$tag" \
    --size "$size" --objective "$obj" --steps "$steps" "$@" \
    2>&1 | tee "$LOG/$tag.train.log" | grep -E "PixelDiT|VAL |EARLY|done|Error|Traceback" | tail -40
}

eval_latent() {
  local tag=$1 vae=$2
  [[ -f "$OUT/$tag/ckpt_best.pt" ]] || { echo "  no ckpt for $tag"; return; }
  if past "$EVAL_DEADLINE_S"; then
    say "SKIP EVAL $tag — past ${EVAL_DEADLINE_H}h (elapsed $(elapsed_h)h)"
    return 1
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
    say "SKIP EVAL $tag — past ${EVAL_DEADLINE_H}h (elapsed $(elapsed_h)h)"
    return 1
  fi
  say "EVAL $tag  elapsed=$(elapsed_h)h"
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/$tag/ckpt_best.pt" --mode pixel "$@" \
    --tile 64 --pad 16 --tile-batch 8 --steps 8 \
    --sweep-steps 2,4,8,16,32 --sweep-pairs 8 \
    --out "$OUT/$tag/eval_official" 2>&1 | tee "$LOG/$tag.eval.log" | grep -E "OFFICIAL|SWEEP|FAIL" | tail -10
}

# ---------------------------------------------------------------- 0. setup
say "SETUP"
"$PY" -m diffusion.smoke_test 2>&1 | tail -2
for v in flux1-vae sd-vae-ft-ema sdxl-vae; do
  n=$(ls "data/latents/$v"/*.pt 2>/dev/null | wc -l)
  if [[ "$n" -lt 400 ]]; then
    echo "  precomputing latents: $v"
    "$PY" -m diffusion.precompute_latents --data-root "$DATA" --vae "$v" --out "data/latents/$v" \
      2>&1 | tail -2
  else
    echo "  latents ok: $v ($n)"
  fi
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ------------------------------------------- 1/2. core: latent flow vs reg (Flux VAE)
train_latent latent_flow_flux flux1-vae S flow 22000 $LAT
eval_latent  latent_flow_flux flux1-vae

train_latent latent_reg_flux flux1-vae S reg 22000 $LAT
eval_latent  latent_reg_flux flux1-vae

# ------------------------------------------- 3/4. core: pixel flow vs reg
train_pixel  pixel_reg S reg 16000 $PIX
eval_pixel   pixel_reg --objective reg

train_pixel  pixel_flow S flow 16000 $PIX
eval_pixel   pixel_flow --objective flow --residual

# ------------------------------------------- 5/6. param-matched (XS ~5.7M vs SwinIR 4.05M)
train_latent latent_flow_flux_XS flux1-vae XS flow 22000 $LAT_XS
eval_latent  latent_flow_flux_XS flux1-vae

train_pixel  pixel_reg_XS XS reg 14000 $PIX_XS
eval_pixel   pixel_reg_XS --objective reg

# ------------------------------------------- 7/8. VAE ablation (4ch vs 16ch)
train_latent latent_flow_sd sd-vae-ft-ema S flow 16000 $LAT
eval_latent  latent_flow_sd sd-vae-ft-ema

train_latent latent_flow_sdxl sdxl-vae S flow 13000 $LAT
eval_latent  latent_flow_sdxl sdxl-vae

# ---------------------------------------------------------------- summary
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
