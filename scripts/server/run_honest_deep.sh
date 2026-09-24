#!/usr/bin/env bash
# honest_deep = the model-side honest items, chained after honest48 (waits for it).
# Two levers that need no GPU beyond the arms:
#   * --dwt-shift 4 : shift-ensemble (DTCWT-like) wavelet-HF loss -> the net can no
#     longer dodge detail into the wrong Haar subband; the biggest cheap win candidate.
#   * --dwt-dual    : WaveletDualUNet (Direction 1 done properly): LL through a full
#     U-Net at half res (lossless stride-2 receptive field) + HL/LH/HH shallow branch.
# Zero-pretrain arms (ZR) are clean arch ablations vs the s1_b64 anchor / wave-lambda5.
# Pretrain arms (PR, init=b64_ft15k, Muon5e-3) stack onto the delivery champion.
# Sample budget ~1.28M; judge SSIM/MUSIQ/MANIQA; +-0.05 tie; no grad-ckpt.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
CHAMP="$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
OUT="$ROOT/experiments/diffusion/honest_deep"
mkdir -p "$OUT" "$LOG"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 240); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }
CORE="--backbone unet --size S --native-lr 0 --objective reg --residual 1 --mult 1,2,4,4 --num-res 2 \
  --decoded-manifest data/decoded/manifest.json --lr-patch 64 --amp --num-workers 12 --cache-data 0 \
  --warmup 500 --ema 0.999 --eval-every 500 --val-pairs 16 --patience 15 --val-eval-steps 8 --save-every 2500"

echo "[honest_deep $(date +%m-%d\ %H:%M)] waiting for honest48 to finish..."
while tmux has-session -t honest48 2>/dev/null; do sleep 30; done
wait_gpu
echo "[honest_deep $(date +%H:%M)] honest48 done -> starting"

run() {  # $1 sub(zero|pretrain) $2 name $3 batch $4 steps, rest=extra
  local SUB="$1" NAME="$2" BATCH="$3" STEPS="$4"; shift 4
  local DIR="$OUT/$SUB/$NAME"
  if [ -f "$DIR/eval_iqa/eval.json" ]; then say "SKIP $SUB/$NAME"; return 0; fi
  local OPT
  if [ "$SUB" = pretrain ]; then OPT="--init $CHAMP --optimizer muon --muon-lr 5e-3"; else OPT="--optimizer adamw --lr 3e-4"; fi
  wait_gpu
  say "TRAIN $SUB/$NAME (b=$BATCH s=$STEPS | $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" $CORE $OPT \
    --base 64 --batch "$BATCH" --steps "$STEPS" --dwt-loss --dwt-weight 8 --dwt-levels 2 "$@" \
    > "$LOG/hd_${SUB}_${NAME}.train.log" 2>&1
  grep -E "DWT-DUAL|optimizer=|VAL |done|Error|Traceback|nan" "$LOG/hd_${SUB}_${NAME}.train.log" | tail -8
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --out "$DIR/eval_iqa" > "$LOG/hd_${SUB}_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/hd_${SUB}_${NAME}.eval.log" | tail -2
}

# ---- zero-pretrain (clean arch / loss ablations) ----
run zero z_shift4       128 10000 --dwt-shift 4
run zero z_db2shift4    128 10000 --dwt-basis db2 --dwt-shift 4
run zero z_shift4_l12   128 10000 --dwt-shift 4 --dwt-weight 12
run zero z_shift4_lv3   128 10000 --dwt-shift 4 --dwt-levels 3
run zero z_aniso_shift  128 10000 --dwt-basis db2 --dwt-shift 4 --dwt-w-hl 1.5 --dwt-w-lh 0.8
run zero z_dual          128 10000 --dwt-dual
run zero z_dual_shift4   128 10000 --dwt-dual --dwt-shift 4
run zero z_dual_db2      128 10000 --dwt-dual --dwt-basis db2 --dwt-shift 4
run zero z_dual_b96       96 13333 --dwt-dual --base 96
run zero z_dual_b128      64 20000 --dwt-dual --base 128 --min-steps 5000
# ---- pretrain + Muon (stack onto the delivery champion) ----
run pretrain p_shift4     128 10000 --dwt-shift 4
run pretrain p_db2shift4  128 10000 --dwt-basis db2 --dwt-shift 4
run pretrain p_shift4_lv3 128 10000 --dwt-shift 4 --dwt-levels 3
run pretrain p_aniso_shift 128 10000 --dwt-basis db2 --dwt-shift 4 --dwt-w-hl 1.5 --dwt-w-lh 0.8
say "HONEST_DEEP DONE"
