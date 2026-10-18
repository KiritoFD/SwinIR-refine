#!/usr/bin/env bash
# Pre-flight for the OFFICIAL-architecture arms:
#  (1) build all four from the vendored official code -> params + LR->HR shape
#  (2) measure real s/step + peak memory at the largest batch that fits, and project the
#      wall clock for a given sample budget BEFORE scheduling anything
#  (3) prove the new net-input=lr path end-to-end: train a few steps, save, tiled-eval
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find diffusion scripts/server -name '*.py' -o -name '*.sh' 2>/dev/null | xargs -r sed -i 's/\r$//'

echo "===== (1) official architectures ====="
$P -u -m diffusion.baselines_official 2>&1 | grep -vE 'UserWarning|FutureWarning|warnings.warn|return _VF'

echo; echo "===== (2) measured cost per arm (projected hours for 1.28M and 0.32M samples) ====="
for B in edsr_off rcan_off swinir_off hat_off; do
  for CAND in 64 32 16 8; do
    LOG=experiments/diffusion/logs/probeoff_${B}_${CAND}.log
    mkdir -p experiments/diffusion/logs
    $P -u -m diffusion.train_pixel --data-root "$DATA" --out experiments/official_probe \
      --backbone $B --net-input lr --size S --scale 2 --objective reg --lr-patch 64 \
      --batch $CAND --steps 40 --eval-every 0 --warmup 10 --amp --num-workers 12 --cache-data 0 \
      --decoded-manifest data/decoded/manifest.json > $LOG 2>&1
    if grep -qE "out of memory|OutOfMemory" $LOG; then echo "$B batch=$CAND OOM"; continue; fi
    $P - "$B" "$CAND" "$LOG" <<'PY'
import re, sys
name, bs, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
txt = open(log, errors="ignore").read()
sec = [float(x) for x in re.findall(r"\|\s*([0-9.]+)s\s*$", txt, re.M)]
mem = [float(x) for x in re.findall(r"\|\s*([0-9.]+)GB", txt)]
par = re.search(r"params=([0-9.]+)M", txt)
step = [s for s in sec if s > 0]
if len(step) < 2:
    print(f"{name}: could not parse timing; tail: {txt.strip().splitlines()[-1][:160]}"); raise SystemExit
d = step[-1] - step[0]                      # cumulative seconds -> last-step delta
per = max(d, step[-1] / max(len(step), 1))
# use a conservative per-step estimate from the last logged delta
sps = step[-1] / max(len(step), 1)
sps = max(per, sps * 0.6)
for budget in (1_280_000, 320_000):
    iters = budget / bs
    print("%-10s batch=%-3d params=%sM  ~%.3fs/iter -> %6.1f h for %s samples"
          % (name, bs, par.group(1) if par else "?", sps, sps * iters / 3600, f"{budget:,}"))
print("   peak %s GB, aligned batch chosen=%d" % (f"{max(mem):.1f}" if mem else "?", bs))
PY
    break
  done
done

echo; echo "===== (3) end-to-end net-input=lr path (train -> ckpt -> tiled eval) ====="
$P -u -m diffusion.train_pixel --data-root "$DATA" --out experiments/official_smoke \
  --backbone edsr_off --net-input lr --size S --scale 2 --objective reg --lr-patch 64 \
  --batch 8 --steps 60 --eval-every 30 --val-pairs 2 --val-crop 256 --val-eval-steps 8 \
  --warmup 10 --min-steps 1 --patience 999999 --ema 0.999 --amp --num-workers 12 --cache-data 0 \
  --decoded-manifest data/decoded/manifest.json 2>&1 | grep -E "EDSR-official|VAL |done|Error|Traceback" | tail -6
$P -u -m diffusion.eval_official --data-root "$DATA" --ckpt experiments/official_smoke/ckpt_last.pt \
  --mode pixel --objective reg --max-pairs 2 --tile 64 --pad 16 --tile-batch 8 --steps 8 \
  --out experiments/official_smoke/eval_check 2>&1 | grep -E "OFFICIAL|net_input|FAIL|Error|Traceback" | tail -5
rm -rf experiments/official_smoke experiments/official_probe
echo "DONE"
