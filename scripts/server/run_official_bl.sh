#!/usr/bin/env bash
# OFFICIAL-architecture baseline campaign - SERIAL, single GPU, budgets from measurement.
#
# Reads experiments/diffusion/batch_probe.tsv (produced by scripts/server/probe_batches.sh),
# so every arm runs at the batch this 4090 actually measured as best-throughput-within-40GB,
# with steps derived from the shared sample budget (batch x steps = 1.28M = our champion's).
#
# Official = the architecture (loaded from third_party/, the papers' own x2 configs; params
# printed at build time and recorded in the probe table). NOT official = the data (RealSR
# Train 390 pairs, per the user's decision) and the shared recipe, which is deliberately
# AdamW 1e-4 + plain L1: every paper in this ladder trains with Adam at 1e-4, and using our
# champion's Muon 5e-3 + wavelet loss for the baselines would be a different experiment.
# `unet_ctrl` is our own U-Net under that same recipe so the architecture comparison is
# like-for-like; our delivered champion stays as its own row.
set -uo pipefail
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
source scripts/server/bl_arms.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find diffusion scripts/server -name '*.py' -o -name '*.sh' 2>/dev/null | xargs -r sed -i 's/\r$//'
LOG=experiments/diffusion/logs; OUT=experiments/diffusion/baselines_official
TSV=experiments/diffusion/batch_probe.tsv
mkdir -p "$LOG" "$OUT"
say(){ echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu(){ for _ in $(seq 1 480); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }

say "PHASE 0: tiling-correctness gate (whole-image vs tiled must agree, all official arms)"
bash scripts/server/tiling_gate.sh > experiments/tiling_gate.log 2>&1 || true
grep -E "OK|TILING BUG|FAILED" experiments/tiling_gate.log
if grep -qE "TILING BUG|FAILED" experiments/tiling_gate.log; then
  echo "GATE FAILED - refusing to spend GPU hours on a possibly mis-tilled evaluator"; exit 1
fi

[ -s "$TSV" ] || { echo "missing $TSV - run scripts/server/probe_batches.sh first"; exit 1; }

ev(){ # dir
  local DIR="$1" CK="$DIR/ckpt_best.pt"
  [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  if [ ! -f "$DIR/eval_iqa/eval.json" ]; then
    wait_gpu; say "EVAL $(basename $DIR) (A-strict pad16 + IQA)"
    "$P" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 4 --iqa --out "$DIR/eval_iqa" > "$LOG/bo_$(basename $DIR).eval.log" 2>&1
    grep -E "OFFICIAL|FAIL|Error" "$LOG/bo_$(basename $DIR).eval.log" | tail -2
  fi
  if [ ! -f "$DIR/eval_iqa_tta/eval.json" ]; then
    wait_gpu; say "EVAL-TTA $(basename $DIR) (8-fold D4)"
    "$P" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 4 --iqa --tta --out "$DIR/eval_iqa_tta" > "$LOG/bo_$(basename $DIR).tta.log" 2>&1
    grep -E "OFFICIAL|FAIL|Error" "$LOG/bo_$(basename $DIR).tta.log" | tail -2
  fi
}

train(){ # arm batch steps
  local A="$1" BS="$2" ST="$3"
  local DIR="$OUT/$A"
  [ -f "$DIR/eval_iqa/eval.json" ] && { say "SKIP $A (already evaluated)"; ev "$DIR"; return 0; }
  wait_gpu; say "TRAIN $A batch=$BS steps=$ST samples=$(( BS * ST ))"
  "$P" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" \
    --scale 2 --objective reg --lr-patch 64 --batch "$BS" --steps "$ST" \
    --lr 1e-4 --warmup 500 --ema 0.999 --amp --num-workers 12 --cache-data 0 \
    --decoded-manifest data/decoded/manifest.json --eval-every 2000 --val-pairs 16 --val-crop 256 \
    --val-eval-steps 8 --patience 20 --min-steps 8000 --save-every 10000 \
    $(flags_for "$A") > "$LOG/bo_${A}.train.log" 2>&1
  grep -E "official|params=|VAL 2|done|Error|Traceback|nan|out of memory" "$LOG/bo_${A}.train.log" | tail -8
  ev "$DIR"
}

say "PHASE 1: arms ordered by MEASURED hours (cheap first, so the table fills early)"
# The probe table's own order is the arm list's order, which would put a 40 h
# transformer fourth and hold up everything behind it. Sort by the measured hours column.
"$P" - "$TSV" > "$TSV.ordered" <<'PY'
import sys
lines = open(sys.argv[1]).read().splitlines()
if lines:
    head, rest = lines[0], [l for l in lines[1:] if l.strip()]
    def hrs(l):
        p = l.split("\t")
        try:
            return float(p[7])
        except Exception:
            return 1e9
    print("\n".join(sorted(rest, key=hrs)))
PY
while IFS=$'\t' read -r A BS ST PER PEAK PAR THR HRS NOTE; do
  [ "$A" = "arm" ] && continue                                  # header
  [ -z "${BS:-}" ] && continue
  echo "-- plan $A: batch=$BS steps=$ST ${PER}s/iter peak=${PEAK}GB params=${PAR}M -> ${HRS}h $NOTE"
  train "$A" "$BS" "$ST"
done < "$TSV.ordered"

say "PHASE 2: summary for the table"
"$P" - <<'PY'
import json, glob
def row(tag, p):
    o = json.load(open(p)); iq = o.get("iqa_scores") or {}
    print("%-30s | %-14s | pad=%-3s Y=%7.4f SSIM=%.5f MUSIQ=%6.2f MANIQA=%.5f%s" % (
        tag, "TTA" if "tta" in p else "no-TTA", o.get("pad"),
        o.get("psnr_y", float("nan")), o.get("ssim_y", float("nan")),
        iq.get("musiq", float("nan")), iq.get("maniqa", float("nan")),
        "" if o.get("pad") == 16 else "   <-- pad!=16"))
print("===== OFFICIAL arms =====")
for p in sorted(glob.glob("experiments/diffusion/baselines_official/*/eval_iqa*/eval.json")):
    row(p.split("baselines_official/")[1].split("/")[0], p)
print("===== earlier pre-upsampling VARIANT arms (separate, labelled block) =====")
for p in sorted(glob.glob("experiments/diffusion/baselines/*/eval_iqa*/eval.json")):
    row(p.split("baselines/")[1].split("/")[0], p)
PY
say "OFFICIAL-BASELINE CAMPAIGN DONE"
