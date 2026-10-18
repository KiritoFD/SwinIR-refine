#!/usr/bin/env bash
# Measure the right batch for every arm of the official ladder, on this box, by running
# each arm briefly at each candidate batch size. Selection rule = best THROUGHPUT among
# the batches that stay within MEM_CAP_GB, not "the biggest that does not OOM"
# (docs/INFRA.md: this card slows super-linearly past ~40 GB, and batch never buys
# throughput once the kernels are saturated: 940 -> 1067 samples/s then flat).
#
# Output: experiments/diffusion/batch_probe.tsv
#   arm  batch  steps  sec_per_iter  peak_gb  params_M  img_per_s  hours_for_budget  note
# consumed by scripts/server/run_official_bl.sh.
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
source scripts/server/bl_arms.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=experiments/diffusion/logs; RAW=experiments/diffusion/batch_probe${PROBE_SUFFIX:+_$PROBE_SUFFIX}.raw
TSV=experiments/diffusion/batch_probe${PROBE_SUFFIX:+_$PROBE_SUFFIX}.tsv
mkdir -p "$LOG"
: > "$RAW"
STEPS=$(probe_secs)

for A in $ARMS; do
  FL=$(flags_for "$A")
  for CAND in $(batch_candidates "$A"); do
    L="$LOG/pb_${A}_${CAND}.log"
    echo "[probe] $A batch=$CAND ..."
    "$P" -u -m diffusion.train_pixel --data-root "$DATA" --out experiments/batch_probe \
      --scale 2 --objective reg --lr-patch 64 --batch "$CAND" --steps "$STEPS" --eval-every 0 \
      --warmup 20 --amp --num-workers 12 --cache-data 0 --decoded-manifest data/decoded/manifest.json \
      $FL > "$L" 2>&1
    if grep -qE "out of memory|OutOfMemory" "$L"; then
      echo "$A $CAND OOM OOM 0 0 0" >> "$RAW"; echo "  -> OOM"; continue
    fi
    "$P" - "$A" "$CAND" "$L" >> "$RAW" <<'PY'
import re, sys
arm, cand, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
txt = open(path, errors="ignore").read()
marks = re.findall(r"step (\d+)/\d+ \|[^|]*\|[^|]*\|\s*([0-9.]+)GB \| (\d+)s", txt)
par = re.search(r"params=([0-9.]+)M", txt)
if len(marks) >= 3:
    (s0, _, t0), (s1, _, t1) = marks[-2], marks[-1]
    per = (int(t1) - int(t0)) / max(int(s1) - int(s0), 1)
    peak = max(float(m[1]) for m in marks)
    print("%s %d %.4f %.1f %s 0" % (arm, cand, per, peak, par.group(1) if par else "0"))
else:
    print("%s %d NA NA %s 1" % (arm, cand, par.group(1) if par else "0"))     # ran, unparseable
PY
  done
  echo "  [probe] $A done"
done
rm -rf experiments/batch_probe

echo; echo "===== pick best-throughput batch per arm ====="
"$P" - "$RAW" "$TSV" "$BUDGET" "$MEM_CAP_GB" <<'PY'
import sys
raw, tsv, budget, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
rows = {}
for ln in open(raw):
    f = ln.split()
    if len(f) != 6:
        continue
    arm, cand, per, peak, par, bad = f
    if per == "OOM":
        rows.setdefault(arm, []).append((int(cand), None, None, par, "OOM"))
        continue
    if per == "NA":
        rows.setdefault(arm, []).append((int(cand), None, None, par, "unparsable"))
        continue
    per = float(per); peak = float(peak)
    rows.setdefault(arm, []).append((int(cand), per, peak, par, ""))

out = open(tsv, "w")
out.write("arm\tbatch\tsteps\tsec_per_iter\tpeak_gb\tparams_M\timg_per_s\thours\tnote\n")
print("%-12s %5s %7s %12s %8s %9s %9s %7s" % ("arm", "batch", "steps", "s/iter", "peakGB", "paramsM", "img/s", "hours"))
for arm, cands in rows.items():
    ok = [(c, p, k) for c, p, k, _, note in cands if p and k and k <= float(cap)]
    note = ""
    if not ok:
        ok = [(c, p, k) for c, p, k, _, note2 in cands if p and k]
        note = "no batch within cap"
        if not ok:
            print("%-12s  NOTHING FITTED - skip this arm" % arm); continue
    c, p, k = max(ok, key=lambda t: t[0] / t[1])          # best throughput = samples/s
    par = next((x[3] for x in cands if x[0] == c), "0")
    steps = int(round(budget / c))
    hours = p * steps / 3600.0
    print("%-12s %5d %7d %12.3f %8.1f %9s %9.1f %7.1f  %s" % (arm, c, steps, p, k, par, c / p, hours, note))
    out.write("%s\t%d\t%d\t%.4f\t%.1f\t%s\t%.2f\t%.1f\t%s\n" % (arm, c, steps, p, k, par, c / p, hours, note))
out.close()
print("\nwrote", tsv)
PY

# A single-arm re-run (PROBE_SUFFIX set) must not lose the others: fold the fresh row(s)
# into the main table, replacing that arm's old row.
if [ -n "${PROBE_SUFFIX:-}" ] && [ -f experiments/diffusion/batch_probe.tsv ]; then
  "$P" - experiments/diffusion/batch_probe.tsv "$TSV" <<'PY'
import sys
main, upd = sys.argv[1], sys.argv[2]
mh = open(main).read().splitlines()
uh = open(upd).read().splitlines()
if mh and uh:
    upd_arms = {l.split("\t")[0] for l in uh[1:] if l.strip()}
    keep = [l for l in mh[1:] if l.split("\t")[0] not in upd_arms]
    open(main, "w").write("\n".join([mh[0]] + keep + uh[1:]) + "\n")
    print("merged", sorted(upd_arms), "into", main)
PY
fi
