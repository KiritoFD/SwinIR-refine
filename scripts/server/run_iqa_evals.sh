#!/usr/bin/env bash
# Re-run the official 100-pair eval with MUSIQ / MANIQA for the arms that matter.
#
# These are no-reference perceptual metrics, so they need the SR image and
# nothing else -- but the SR image is not saved, so the whole eval has to be
# redone (it is deterministic, seed 1234, so the PSNR/SSIM come out identical
# to the earlier run and the two tables join cleanly).
#
# Serial on purpose: each eval wants most of the GPU.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

# name -> "ckpt_path mode objective"
ARMS="
s1_b64:$ROOT/experiments/diffusion/s1_sweep/b64_finetune/ckpt_best.pt:pixel:reg
s1_b48:$ROOT/experiments/diffusion/s1_sweep/b48_finetune/ckpt_best.pt:pixel:reg
s1_b40:$ROOT/experiments/diffusion/s1_sweep/b40_finetune/ckpt_best.pt:pixel:reg
s1_b32:$ROOT/experiments/diffusion/s1_sweep/b32_finetune/ckpt_best.pt:pixel:reg
dit_pixel_reg:$ROOT/experiments/diffusion/pixel_reg/ckpt_best.pt:pixel:reg
dit_pixel_reg_XS:$ROOT/experiments/diffusion/pixel_reg_XS/ckpt_best.pt:pixel:reg
"

for line in $ARMS; do
  [ -z "$line" ] && continue
  IFS=: read -r NAME CKPT MODE OBJ <<< "$line"
  if [ ! -f "$CKPT" ]; then
    say "SKIP $NAME -- no ckpt at $CKPT"
    continue
  fi
  OUT="$ROOT/experiments/diffusion/iqa_eval/$NAME"
  if [ -f "$OUT/eval.json" ]; then
    say "SKIP $NAME -- already have $OUT/eval.json"
    continue
  fi
  say "IQA EVAL $NAME  ($MODE / $OBJ)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CKPT" --mode "$MODE" --objective "$OBJ" \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT" > "$LOG/${NAME}_iqa.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/${NAME}_iqa.log" | tail -3
done

say "IQA EVALS DONE"
