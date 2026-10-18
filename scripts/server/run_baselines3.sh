#!/usr/bin/env bash
# Baseline campaign #3: (A) re-evaluate the 4 finished baselines at pad 16 so the whole
# A-strict ladder shares one evaluation protocol (they were run at pad 8 while every
# champion arm uses pad 16); (B) train + eval the remaining classic arms and three
# modern top-conference arms (RLFN NTIRE'22 / SRFormer ICCV'23 / HAT CVPR'23) plus a
# fair re-run of SRResNet (its first run collapsed: VAL peaked at 5k then fell to 20.9).
#
# Budget discipline (see docs/INFRA.md): one sample budget for every baseline arm,
# batch x steps = 1.28M (same as the existing EDSR/RCAN/SRResNet/RRDB arms and equal to
# our 128 x 10000). Batch is probed per arm on the 4090; steps follow from it.
set -uo pipefail
cd /home/ds/realsr || exit 1
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find scripts/server diffusion -name '*.sh' -o -name '*.py' 2>/dev/null | xargs -r sed -i 's/\r$//'
LOG=experiments/diffusion/logs
OUT=experiments/diffusion/baselines
mkdir -p "$LOG" "$OUT"
BUDGET=1280000
say(){ echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu(){ for _ in $(seq 1 480); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }

ev(){ # ckpt dir  -> pad16 non-TTA + TTA, both idempotent
  local CK="$1" DIR="$2"
  if [ ! -f "$DIR/eval_iqa16/eval.json" ]; then
    wait_gpu; say "EVAL16 $(basename $DIR)"
    "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --out "$DIR/eval_iqa16" > "$LOG/ev16_$(basename $DIR).log" 2>&1
    grep -E "OFFICIAL|FAIL|Error" "$LOG/ev16_$(basename $DIR).log" | tail -2
  fi
  if [ ! -f "$DIR/eval_iqa16_tta/eval.json" ]; then
    wait_gpu; say "EVAL16-TTA $(basename $DIR)"
    "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --out "$DIR/eval_iqa16_tta" > "$LOG/ev16t_$(basename $DIR).log" 2>&1
    grep -E "OFFICIAL|FAIL|Error" "$LOG/ev16t_$(basename $DIR).log" | tail -2
  fi
}

say "PHASE A: pad-16 re-evaluation of finished arms"
for B in edsr rcan srresnet rrdb; do
  CK="$OUT/${B}_x2/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/${B}_x2/ckpt_last.pt"
  [ -f "$CK" ] && ev "$CK" "$OUT/${B}_x2"
done

probe(){ # name extra-flags -> echo a batch that fits (64/32/16/8)
  local N="$1"; shift
  for CAND in 64 32 16 8; do
    wait_gpu
    "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT/_probe_$N" \
      --scale 2 --objective reg --lr-patch 64 --batch "$CAND" --steps 6 --eval-every 0 \
      --amp --num-workers 8 --cache-data 0 "$@" > "$LOG/probe_${N}_${CAND}.log" 2>&1
    if grep -qE "out of memory|OutOfMemory" "$LOG/probe_${N}_${CAND}.log"; then
      echo "  probe $N batch=$CAND OOM" >&2; rm -rf "$OUT/_probe_$N"; continue
    fi
    rm -rf "$OUT/_probe_$N"; echo "$CAND"; return 0
  done
  echo 8
}

arm(){ # name extra-flags
  local N="$1"; shift
  local DIR="$OUT/${N}_x2"
  if [ -f "$DIR/eval_iqa16/eval.json" ]; then say "SKIP $N"; return 0; fi
  say "PROBE+TRAIN $N ($*)"
  local BS; BS=$(probe "$N" "$@"); BS=${BS:-64}
  local ST; ST=$(( BUDGET / BS ))
  wait_gpu; say "TRAIN $N batch=$BS steps=$ST (samples=$BS x $ST = $BUDGET)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" \
    --size S --scale 2 --objective reg --lr-patch 64 --batch "$BS" --steps "$ST" \
    --lr 3e-4 --warmup 500 --ema 0.999 --amp --num-workers 12 --cache-data 0 \
    --eval-every 1000 --val-pairs 16 --patience 20 --min-steps 4000 --val-eval-steps 8 \
    --save-every 5000 "$@" > "$LOG/bl3_${N}.train.log" 2>&1
  grep -E "align=|done|Error|Traceback|nan|out of memory" "$LOG/bl3_${N}.train.log" | tail -8
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  ev "$CK" "$DIR"
}

say "PHASE B: new arms (cheap first so the table fills early)"
# SRCNN predicts the image itself -> --residual 0 (identity-initialised head, see baselines_sr.py)
# HAT/SRFormer run with --eval-every 0: the in-training val does one UNTILED full-image
# forward (train_pixel.run_val), which is fine for the CNNs but would need ~5.6M tokens
# for a window transformer -> OOM / minutes per val. Those two arms use ckpt_last.
arm srcnn   --backbone srcnn --residual 0
arm rdn     --backbone rdn
arm vdsr    --backbone vdsr --base 64
arm rlfn    --backbone rlfn
arm srresnet_v2 --backbone srresnet --base 64 --lr 1e-4
arm hat     --backbone hat --eval-every 0
arm srformer --backbone srformer --eval-every 0

say "PHASE C: summary"
$PY - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("experiments/diffusion/baselines/*_x2/eval*/eval.json")):
    o = json.load(open(p)); iq = o.get("iqa_scores") or {}
    rows.append((p.replace("experiments/diffusion/baselines/", ""),
                 o.get("psnr_y"), o.get("ssim_y"), iq.get("musiq"), iq.get("maniqa"), o.get("pad")))
for r in rows:
    print("%-52s Y=%s SSIM=%s MUSIQ=%s MANIQA=%s pad=%s" % tuple(
        r[:1] + [("%.4f" % v if isinstance(v, float) else v) for v in r[1:]]))
print("rows:", len(rows))
PY
say "BASELINES-3 DONE"
