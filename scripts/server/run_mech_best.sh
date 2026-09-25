#!/usr/bin/env bash
# Corrected mech phase: the Phase-M sweep showed every deviation is WORSE than the
# base Muon lr=5e-3/mom0.95/ns5 (= champmuon), but the auto-selector (bug) picked a
# handicapped 2e-3 for mech.  Re-run the promising mechanism arms at the CORRECT best
# Muon (5e-3).  N3 adversarial (data lever) is the headliner -- its 2e-3 run already
# gave the highest MANIQA of the whole project; give it a fair 5e-3 shot.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
CHAMP="$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt"
PRE="data/pretrain/DIV2K_train_HR,data/pretrain/Flickr2K"
PREVAL="data/pretrain/DIV2K_valid_HR"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"; OUT="$ROOT/experiments/diffusion/mech_best"; mkdir -p "$OUT" "$LOG"
MU="--optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 240); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }
te() {  # $1 dir, rest flags
  local DIR="$1"; shift
  if [ -f "$DIR/eval_iqa/eval.json" ]; then say "SKIP $(basename $DIR)"; return 0; fi
  wait_gpu; say "TRAIN $(basename $DIR) ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
    --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --batch 128 --amp \
    --num-workers 12 --cache-data 0 --warmup 500 --steps 10000 --ema 0.999 --dwt-loss --dwt-weight 8 \
    --dwt-levels 2 --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 \
    --save-every 2500 "$@" > "$LOG/mb_$(basename $DIR).train.log" 2>&1
  grep -E "optimizer=|adversary|init weights|done|Error|Traceback|nan" "$LOG/mb_$(basename $DIR).train.log" | tail -6
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --out "$DIR/eval_iqa" > "$LOG/mb_$(basename $DIR).eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/mb_$(basename $DIR).eval.log" | tail -2
}
# reference to beat = champmuon (5e-3, lam8): SSIM .92797 MUSIQ 55.85 MANIQA .35206
te "$OUT/p_shift4"  $MU --init "$CHAMP" --dwt-shift 4
te "$OUT/p_aniso"  $MU --init "$CHAMP" --dwt-w-hl 1.5 --dwt-w-lh 0.8 --dwt-shift 4
te "$OUT/p_dtcwt"  $MU --init "$CHAMP" --dwt-basis dtcwt
# N3 adversarial pretrain(Muon5e-3) -> ft(Muon5e-3) -> TTA, the data-lever headliner
ADV="$OUT/n3_pretrain"; FT="$OUT/n3_ft"
if [ ! -f "$ADV/ckpt_last.pt" ] && [ ! -f "$ADV/ckpt_best.pt" ]; then
  wait_gpu; say "N3 pretrain adversarial on L1+wavelet-HF, Muon5e-3"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$ADV" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
    --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --batch 128 --amp \
    --num-workers 12 --cache-data 0 --warmup 500 --steps 25000 --ema 0.999 --dwt-loss --dwt-weight 8 \
    --dwt-levels 2 --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 \
    --save-every 2500 $MU --adv-deg --pretrain-root "$PRE" --pretrain-val-root "$PREVAL" \
    > "$LOG/mb_n3_pretrain.train.log" 2>&1
  grep -E "adversary|done|Error|Traceback|nan" "$LOG/mb_n3_pretrain.train.log" | tail -6
fi
ACK="$ADV/ckpt_best.pt"; [ -f "$ACK" ] || ACK="$ADV/ckpt_last.pt"
if [ -f "$ACK" ] && [ ! -f "$FT/eval_iqa/eval.json" ]; then
  te "$FT" $MU --init "$ACK"
  CK="$FT/ckpt_best.pt"; [ -f "$CK" ] || CK="$FT/ckpt_last.pt"
  wait_gpu
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --out "$FT/eval_iqa_tta" > "$LOG/mb_n3_tta.log" 2>&1
  grep -E "OFFICIAL|Error" "$LOG/mb_n3_tta.log" | tail -1
fi
say "MECH-BEST DONE (all at Muon 5e-3)"
