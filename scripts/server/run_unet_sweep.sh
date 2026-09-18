#!/usr/bin/env bash
# Five U-Net sizes: BSRGAN pretrain -> RealSR fine-tune -> official eval, serially.
#
# The step counts are derived from a SAMPLE budget, not fixed per model.  That
# is the whole point of the sweep: a 41M net and a 5M net must see the same
# number of training samples or the comparison is meaningless.
#
#   PRETRAIN_SAMPLES = 1,000,000   (282 epochs over the 3550 pretrain images)
#   FINETUNE_SAMPLES = 3,900,000   (= 10000 x 390, the run currently in flight)
#
# Throughput notes that drive the sizing:
#   * BSRGAN loader is capped at ~257 samples/s (12 workers) -- the degradation
#     itself, not the decode.  Every size is loader-bound during pretraining,
#     so all five take the same wall time there.
#   * RealSR loader does ~1100 samples/s, so during fine-tuning the GPU becomes
#     the limit for the bigger nets and the time scales with parameter count.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
MANIFEST="${MANIFEST:-data/decoded/manifest.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

OUT_ROOT="$ROOT/experiments/diffusion/unet_sweep"
LOG="$ROOT/experiments/diffusion/logs"
mkdir -p "$OUT_ROOT" "$LOG"

PRETRAIN_ROOT="${PRETRAIN_ROOT:-data/pretrain/DIV2K_train_HR,data/pretrain/DIV2K_valid_HR,data/pretrain/Flickr2K}"
# Only the big nets get pretrained.  A 5M net on 406 pairs is not data-starved,
# so pretraining buys it little; the 18-41M nets are, and that is where the
# question "does more data fix it" is actually interesting.
PRETRAIN_SIZES="${PRETRAIN_SIZES:-s deep wide}"
# Step counts are identical across sizes and so are the batches, which makes the
# sample budget identical too -- that is what keeps the comparison honest.
PRE_STEPS="${PRE_STEPS:-20000}"
FT_STEPS="${FT_STEPS:-25000}"
PRE_BATCH="${PRE_BATCH:-128}"
FT_BATCH="${FT_BATCH:-256}"
PRE_WORKERS="${PRE_WORKERS:-12}"
FT_WORKERS="${FT_WORKERS:-12}"
PRE_LR="${PRE_LR:-2e-4}"
FT_LR="${FT_LR:-3e-4}"
SIZES="${SIZES:-xs shallow s deep wide}"

ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

cfg() {                       # name -> "base num_res"
  case "$1" in
    xs)      echo "32 2" ;;
    shallow) echo "64 1" ;;
    s)       echo "64 2" ;;
    deep)    echo "64 3" ;;
    wide)    echo "96 2" ;;
    *) echo ""; return 1 ;;
  esac
}

T0=$(date +%s)
for SZ in $SIZES; do
  read -r BASE NR <<< "$(cfg "$SZ")" || { echo "unknown size $SZ"; continue; }
  [[ -z "${BASE:-}" ]] && { echo "unknown size $SZ"; continue; }

  PRE_OUT="$OUT_ROOT/${SZ}_pretrain"
  FT_OUT="$OUT_ROOT/${SZ}_finetune"

  DO_PRE=0
  for X in $PRETRAIN_SIZES; do [[ "$X" == "$SZ" ]] && DO_PRE=1; done
  say "SIZE=$SZ  base=$BASE num_res=$NR  |  pretrain ${DO_PRE} (${PRE_STEPS} x ${PRE_BATCH})  finetune ${FT_STEPS} x ${FT_BATCH}"

  # ------------------------------------------------ stage 1: BSRGAN pretrain
  if (( DO_PRE )) && [[ "${SKIP_PRETRAIN:-0}" != "1" ]]; then
    say "TRAIN ${SZ}_pretrain  (BSRGAN, $(( PRE_STEPS * PRE_BATCH )) samples)"
    "$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$PRE_OUT" \
      --backbone unet --size S --base "$BASE" --num-res "$NR" --native-lr 1 \
      --pretrain-root "$PRETRAIN_ROOT" --decoded-manifest "$MANIFEST" \
      --lr-patch 64 --batch "$PRE_BATCH" --amp --num-workers "$PRE_WORKERS" \
      --cache-data 0 --lr "$PRE_LR" --warmup 500 --steps "$PRE_STEPS" \
      --eval-every $(( PRE_STEPS / 8 )) --val-pairs 16 \
      --patience 100 --min-steps 100000 --val-eval-steps 8 --save-every 5000 \
      2>&1 | tee "$LOG/${SZ}_pretrain.train.log" \
      | grep -E "PixelUNet|native_lr|sampler:|BSRGANDataset|VAL |EARLY|Error|Traceback" | tail -60
  fi

  # ------------------------------------------------ stage 2: RealSR fine-tune
  INIT=""
  if (( DO_PRE )) && [[ -f "$PRE_OUT/ckpt_best.pt" ]]; then
    INIT="--init $PRE_OUT/ckpt_best.pt"
  fi
  say "TRAIN ${SZ}_finetune  (RealSR, $(( FT_STEPS * FT_BATCH )) samples)  init=${INIT:-none}"
  "$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$FT_OUT" \
    --backbone unet --size S --base "$BASE" --num-res "$NR" --native-lr 1 \
    --decoded-manifest "$MANIFEST" $INIT \
    --lr-patch 64 --batch "$FT_BATCH" --amp --num-workers "$FT_WORKERS" \
    --cache-data 0 --lr "$FT_LR" --warmup 500 --steps "$FT_STEPS" \
    --eval-every $(( FT_STEPS / 20 )) --val-pairs 16 \
    --patience 8 --min-steps $(( FT_STEPS / 5 )) --val-eval-steps 8 --save-every 5000 \
    2>&1 | tee "$LOG/${SZ}_finetune.train.log" \
    | grep -E "PixelUNet|native_lr|sampler:|VAL |EARLY|Error|Traceback" | tail -60

  # ------------------------------------------------ stage 3: official eval
  say "EVAL ${SZ}_finetune (official 100 pairs)"
  if [[ -f "$FT_OUT/ckpt_best.pt" ]]; then
    "$PY" -m diffusion.eval_official --data-root "$DATA" \
      --ckpt "$FT_OUT/ckpt_best.pt" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 16 --steps 8 \
      --out "$FT_OUT/eval_official" 2>&1 | tee "$LOG/${SZ}_finetune.eval.log" \
      | grep -E "OFFICIAL|FAIL" | tail -5
  else
    echo "  no ckpt for $SZ"
  fi

  EL=$(( ($(date +%s) - T0) / 60 ))
  say "DONE $SZ   elapsed ${EL} min total"
done

say "UNET SWEEP DONE  total $(( ($(date +%s) - T0) / 60 )) min"
