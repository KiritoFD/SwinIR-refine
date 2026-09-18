#!/usr/bin/env bash
# Stride-1 U-Net series: operate at HR (no pixel shuffle), narrow channels.
#
# Why this series: the stride-2/native-LR design runs the whole net at LR scale
# and lifts the result with a pixel shuffle, which cannot represent HR structure
# the residual actually has.  Stride 1 is what EDSR/SRResNet do.  It costs 2.5x
# the activation memory (138 vs 55 MB/sample) and drops hardware efficiency
# (C=32 gives skinny GEMMs: 4.4 vs 10.7 TFLOPS), so it is slower -- but if the
# quality is higher the trade is worth it.
#
# Batch per size is set to land near 35 GB (measured MB/sample from
# bench_s1_sizes.py).
#
# PRETRAIN: the pretrain set is 3550 images vs RealSR's 390, so its sample
# budget must be larger.  It is capped by the loader, not the GPU: BSRGAN
# degradation tops out at ~265 samples/s no matter how many workers, because
# the bottleneck is the main process's IPC + collate, not the workers.
# 15000 x 256 = 3.84M samples ~= 4.0 h per size.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
MANIFEST="${MANIFEST:-data/decoded/manifest.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/s1_sweep}"
LOG="$ROOT/experiments/diffusion/logs"
mkdir -p "$OUT_ROOT" "$LOG"

# DIV2K_valid is deliberately NOT in the train roots: it is the held-out
# pretrain val, and it must not be seen during training.
PRETRAIN_ROOT="${PRETRAIN_ROOT:-data/pretrain/DIV2K_train_HR,data/pretrain/Flickr2K}"
PRETRAIN_VAL_ROOT="${PRETRAIN_VAL_ROOT:-data/pretrain/DIV2K_valid_HR}"
PRETRAIN_SIZES="${PRETRAIN_SIZES:-b48 b64}"
SIZES="${SIZES:-b32 b40 b48 b64}"
PRE_STEPS="${PRE_STEPS:-15000}"
PRE_BATCH="${PRE_BATCH:-256}"
FT_STEPS="${FT_STEPS:-10000}"
PRE_WORKERS="${PRE_WORKERS:-20}"
FT_WORKERS="${FT_WORKERS:-12}"
PRE_LR="${PRE_LR:-2e-4}"
FT_LR="${FT_LR:-3e-4}"

# kill -9 does not hand VRAM back instantly: the next launch can then OOM on its
# very first step, which looks exactly like a hang (main process stuck in CUDA
# error handling, one loader worker spinning).  Wait for the card to drain first.
for _ in $(seq 1 60); do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( M < 2000 )) && break
  echo "waiting for GPU: ${M} MiB still in use"
  sleep 5
done

ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

cfg() {                       # name -> "base ft_batch pre_batch"
  # The pretrain batch must be sized per model too: stride-1 activations scale
  # with base (b32 138, b48 207, b64 276 MB/sample), and a fixed 256 OOM'd the
  # b48/b64 pretrains outright.
  case "$1" in
    b32) echo "32 256 256" ;;
    b40) echo "40 192 192" ;;
    b48) echo "48 160 160" ;;
    b64) echo "64 128 128" ;;
    *) return 1 ;;
  esac
}

T0=$(date +%s)
for SZ in $SIZES; do
  read -r BASE FT_BATCH SZ_PRE_BATCH <<< "$(cfg "$SZ")" || { echo "unknown size $SZ"; continue; }
  PRE_OUT="$OUT_ROOT/${SZ}_pretrain"
  FT_OUT="$OUT_ROOT/${SZ}_finetune"
  DO_PRE=0
  for X in $PRETRAIN_SIZES; do [[ "$X" == "$SZ" ]] && DO_PRE=1; done

  say "SIZE=$SZ  base=$BASE  ft_batch=$FT_BATCH  pretrain=$DO_PRE"

  # ---------------------------------------------- stage 1: BSRGAN pretrain
  if (( DO_PRE )) && [[ "${SKIP_PRETRAIN:-0}" != "1" ]]; then
    say "TRAIN ${SZ}_pretrain  (BSRGAN, $(( PRE_STEPS * SZ_PRE_BATCH )) samples, batch $SZ_PRE_BATCH)"
    "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$PRE_OUT" \
      --backbone unet --size S --base "$BASE" --mult 1,2,4,4 --num-res 2 --native-lr 0 \
      --objective reg --residual 1 \
      --pretrain-root "$PRETRAIN_ROOT" --pretrain-val-root "$PRETRAIN_VAL_ROOT" \
      --decoded-manifest "$MANIFEST" \
      --lr-patch 64 --batch "$SZ_PRE_BATCH" --amp --num-workers "$PRE_WORKERS" \
      --cache-data 0 --lr "$PRE_LR" --warmup 500 --steps "$PRE_STEPS" \
      --eval-every 1000 --val-pairs 32 \
      --patience 8 --min-steps 2000 --val-eval-steps 8 --save-every 5000 \
      > "$LOG/${SZ}_pretrain.train.log" 2>&1
    grep -E "PixelUNet|native_lr|sampler:|VAL |EARLY|Error|Traceback" \
      "$LOG/${SZ}_pretrain.train.log" | tail -60
  fi

  # ---------------------------------------------- stage 2: RealSR fine-tune
  INIT=""
  if (( DO_PRE )) && [[ -f "$PRE_OUT/ckpt_best.pt" ]]; then
    INIT="--init $PRE_OUT/ckpt_best.pt"
  fi
  say "TRAIN ${SZ}_finetune  (RealSR, $(( FT_STEPS * FT_BATCH )) samples)  init=${INIT:-none}"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$FT_OUT" \
    --backbone unet --size S --base "$BASE" --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 \
    --decoded-manifest "$MANIFEST" $INIT \
    --lr-patch 64 --batch "$FT_BATCH" --amp --num-workers "$FT_WORKERS" \
    --cache-data 0 --lr "$FT_LR" --warmup 500 --steps "$FT_STEPS" \
    --eval-every 500 --val-pairs 16 \
    --patience 8 --min-steps 2000 --val-eval-steps 8 --save-every 5000 \
    > "$LOG/${SZ}_finetune.train.log" 2>&1
  grep -E "PixelUNet|native_lr|sampler:|VAL |EARLY|Error|Traceback" \
    "$LOG/${SZ}_finetune.train.log" | tail -60

  # ---------------------------------------------- stage 3: official eval
  say "EVAL ${SZ}_finetune (official 100 pairs)"
  if [[ -f "$FT_OUT/ckpt_best.pt" ]]; then
    "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
      --ckpt "$FT_OUT/ckpt_best.pt" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 8 --steps 8 \
      --out "$FT_OUT/eval_official" > "$LOG/${SZ}_finetune.eval.log" 2>&1
    grep -E "OFFICIAL|FAIL" "$LOG/${SZ}_finetune.eval.log" | tail -5
  else
    echo "  no ckpt for $SZ"
  fi

  say "DONE $SZ   total elapsed $(( ($(date +%s) - T0) / 60 )) min"
done

say "S1 SWEEP DONE  total $(( ($(date +%s) - T0) / 60 )) min"
