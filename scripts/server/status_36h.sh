#!/usr/bin/env bash
# Progress of the 36h matrix: per-arm step/val/best so far.
set -uo pipefail
ROOT="${ROOT:-/home/ds/realsr}"
OUT="$ROOT/experiments/diffusion"

printf "%-22s %-9s %8s %10s %10s %s\n" ARM STEP BEST_Y TRAIN_LOG LAST_VAL EVAL
for d in "$OUT"/*/; do
  tag=$(basename "$d")
  [[ "$tag" == _verify* || "$tag" == logs || "$tag" == vae_noise ]] && continue
  tl="$d/train_log.jsonl"; vl="$d/val_log.jsonl"
  [[ -f "$tl" || -f "$vl" ]] || continue
  step=$(tail -1 "$tl" 2>/dev/null | python3 -c 'import sys,json;print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo -)
  best=$(python3 - "$vl" <<'PY' 2>/dev/null
import json,sys
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    print(f"{max(r['psnr_y'] for r in rows):.3f}" if rows else "-")
except Exception: print("-")
PY
)
  last=$(tail -1 "$vl" 2>/dev/null | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print(f"{d[\"step\"]}:{d[\"psnr_y\"]:.2f}")' 2>/dev/null || echo -)
  ev="-"
  if [[ -f "$d/eval_official/eval.json" ]]; then
    ev=$(python3 -c "import json;print(f\"{json.load(open('$d/eval_official/eval.json'))['psnr_y']:.3f}\")" 2>/dev/null || echo -)
  fi
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf "%-22s %-9s %8s %10s %10s %s\n" "$tag" "$sz" "$step" "$best" "$last" "$ev"
done
echo
echo "live:  tail -f $OUT/run36h.log      tmux attach -t realsr36h"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
