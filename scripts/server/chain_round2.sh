#!/usr/bin/env bash
# Unattended chain: wait for round-1 (wavearms) to finish, re-validate the
# dwt-unet code on the server GPU with the smoke, and only then launch round-2
# (Direction 1 arms).  If the smoke fails, stop and leave round-2 off.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
cd "$ROOT" || exit 1
say() { echo "[chain $(date +%m-%d\ %H:%M:%S)] $*"; }

say "waiting for round-1 (tmux wavearms) to finish..."
while tmux has-session -t wavearms 2>/dev/null; do sleep 30; done
# let the last eval release the GPU
for _ in $(seq 1 30); do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( M < 2000 )) && break
  sleep 10
done

say "round-1 done; validating dwt-unet via server smoke"
"$PY" -u -m diffusion.smoke_new_arms > experiments/diffusion/logs/wu_smoke.log 2>&1
if grep -q 'SMOKE PASS' experiments/diffusion/logs/wu_smoke.log; then
  say "smoke PASS -> launching round-2 (dwt-unet arms)"
  bash scripts/server/run_dwt_unet_arms.sh > experiments/wave_unet.log 2>&1
  say "round-2 done -> launching round-1b (dwt lambda sweep: 2, 5)"
  bash scripts/server/run_dwt_sweep.sh > experiments/wave_sweep.log 2>&1
  say "round-1b done -> launching round-3 (stack dwt-loss on pretrained champion)"
  bash scripts/server/run_stack_dwt.sh > experiments/stack_dwt.log 2>&1
  say "round-3 done; full overnight chain finished"
else
  say "smoke FAILED -> NOT launching round-2; tail of wu_smoke.log:"
  tail -25 experiments/diffusion/logs/wu_smoke.log
fi
