#!/usr/bin/env bash
# Detached progress snapshooter for the exp16 campaign: appends a compact status
# every 20 min for ~17h so no result is ever missed between manual checks.
cd /home/ds/realsr || exit 1
OUT=experiments/exp16_watch.log
EXP=experiments/diffusion/experiment
: > "$OUT"
for i in $(seq 52); do
  {
    echo "===== $(date +%m-%d\ %H:%M) ====="
    tmux has-session -t exp16 2>/dev/null || echo "SESSION: exp16 gone (campaign finished)"
    echo "-- latest OFFICIAL results --"
    grep -hE "OFFICIAL" experiments/exp16.log 2>/dev/null | tail -8
    echo "-- arms with eval.json --"
    find "$EXP" -name eval.json -path "*eval_iqa/*" 2>/dev/null | sed "s#$EXP/##" | sort
    echo "-- current tail --"
    tail -3 experiments/exp16.log 2>/dev/null
    echo
  } >> "$OUT" 2>&1
  sleep 1200
done
echo "watcher done $(date)" >> "$OUT"
