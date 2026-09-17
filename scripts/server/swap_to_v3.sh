#!/usr/bin/env bash
# Wait for pixel_reg (arm 3) to finish + eval, then swap the rest of the matrix
# over to run_36h3.sh with torch.compile -- but only if compile actually wins.
#
# Rationale: arms 1-3 are already trained; restarting them would burn ~7h.
# Arms 4-8 are still ahead of us, so that is where the compile speedup lands.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

OUT="$ROOT/experiments/diffusion"
MARK="$OUT/pixel_reg/eval_official/eval.json"
WLOG="$OUT/swap_to_v3.log"
BASELINE=0.485   # measured latent-S b=64 eager, bench_batch3.log

log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$WLOG"; }

log "watchdog up; waiting for $MARK"
found=0
for _ in $(seq 1 720); do          # up to 6h
  if [[ -f "$MARK" ]]; then found=1; break; fi
  sleep 30
done
log "marker found=$found"

# let any in-flight eval flush, then stop the v2 driver
sleep 20
tmux kill-session -t realsr36h 2>/dev/null
sleep 10
log "stopped v2 session"

# ---- measure compiled throughput (latent-S b=64, same harness as baseline) ----
rm -rf "$OUT/_cmp"
log "benchmarking compiled latent-S b=64 ..."
"$PY" -m diffusion.train_latent --data-root "$DATA" --out "$OUT/_cmp" \
  --vae flux1-vae --latent-cache data/latents/flux1-vae --size S --objective flow \
  --steps 200 --lr-patch 256 --batch 64 --amp --num-workers 8 --compile \
  --eval-every 100000 --save-every 100000 --val-pairs 4 >"$OUT/_cmp.log" 2>&1
rows=$(grep -E '^step' "$OUT/_cmp.log" | tail -2 |
  sed -n 's/^step 0*\([0-9]*\)\/.*| \([0-9.]*\)GB | \([0-9]*\)s$/\1 \2 \3/p')
rm -rf "$OUT/_cmp"

if [[ -z "$rows" ]]; then
  log "compiled bench FAILED -> keeping eager"; C=0
else
  comp=$(printf '%s\n' "$rows" | awk 'NR==1{n1=$1;s1=$3} NR==2{n2=$1;s2=$3} END{printf "%.3f",(s2-s1)/(n2-n1)}')
  log "compiled s/step=$comp  (eager baseline $BASELINE)"
  C=$(awk -v c="$comp" -v b="$BASELINE" 'BEGIN{print (c>0 && c<b) ? 1 : 0}')
  log "decision COMPILE=$C"
fi

export COMPILE=$C
SESS=realsr36h RUN=run_36h3.sh RUNLOG=run36h3.log bash scripts/server/start_36h.sh | tee -a "$WLOG"
log "v3 launched (COMPILE=$C)"
