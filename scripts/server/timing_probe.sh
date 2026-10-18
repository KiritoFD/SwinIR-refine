#!/usr/bin/env bash
# (A) TILING CORRECTNESS for the new LR-input path: one full-image forward vs the tiled
#     forward must agree. A mis-aligned tiler shows up as a catastrophic number (<20 dB),
#     so this is a hard gate before any baseline number is trusted.
# (B) Honest per-step cost: train_pixel logs CUMULATIVE seconds every 50 steps, so the
#     rate is the slope between the last two marks (step number, seconds).
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/diffusion/logs

echo "===== (A) sr_lr_tiled vs full-image forward ====="
$P - <<'PY'
import torch, torch.nn.functional as F, numpy as np
from diffusion.data import build_index
from diffusion.eval_official import sr_lr_tiled, to_tensor, load_rgb_u8
from diffusion import baselines_official as bo

pairs = build_index("data/RealSR(V3)", ("Canon", "Nikon"), "Test", 2)
lr_path = pairs[0][0]
lr = load_rgb_u8(lr_path)[:448, :448]
print("LR crop", lr.shape, "<-", lr_path.split("RealSR(V3)/")[-1])
for name in ("edsr_off", "rcan_off", "swinir_off", "hat_off"):
    try:
        m = bo.build(name, scale=2).to("cuda").eval()
        x = to_tensor(lr).unsqueeze(0).cuda()
        with torch.no_grad():
            full = m(x, torch.zeros(1, device="cuda")).clamp(0, 1)
        tiled_u8 = sr_lr_tiled(m, lr, 2, 64, 16, "cuda", tile_batch=2, bf16=False)
        tiled = to_tensor(tiled_u8).to(full.device)
        h = min(full.shape[-2], tiled.shape[-2]); w = min(full.shape[-1], tiled.shape[-1])
        mse = F.mse_loss(full[:, :, :h, :w], tiled[:, :, :h, :w]).item()
        psnr = 99.0 if mse <= 1e-12 else -10 * np.log10(mse)
        print("  %-11s full%s vs tiled%s -> %.2f dB  %s"
              % (name, tuple(full.shape), tuple(tiled.shape), psnr,
                 "OK" if psnr > 30 else "*** TILING BUG ***"))
        del m, full, tiled
        torch.cuda.empty_cache()
    except Exception as e:
        print("  %-11s skipped: %s: %s" % (name, type(e).__name__, str(e)[:90]))
PY

echo; echo "===== (B) measured s/iter + projected wall clock per sample budget ====="
for B in edsr_off rcan_off swinir_off hat_off; do
  for CAND in 64 32 16 8 4; do
    LOG=experiments/diffusion/logs/tim_${B}_${CAND}.log
    $P -u -m diffusion.train_pixel --data-root "$DATA" --out experiments/timing_probe \
      --backbone $B --net-input lr --size S --scale 2 --objective reg --lr-patch 64 \
      --batch $CAND --steps 300 --eval-every 0 --warmup 20 --amp --num-workers 12 --cache-data 0 \
      --decoded-manifest data/decoded/manifest.json > $LOG 2>&1
    if grep -qE "out of memory|OutOfMemory" $LOG; then echo "$B  bs=$CAND OOM"; continue; fi
    $P - "$B" "$CAND" "$LOG" <<'PY'
import re, sys
name, bs, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
marks = re.findall(r"step (\d+)/\d+ \|[^|]*\|[^|]*\|\s*([0-9.]+)GB \| (\d+)s", open(log, errors="ignore").read())
if len(marks) >= 3:
    (s0, _, t0), (s1, _, t1) = marks[-2], marks[-1]
    per = (int(t1) - int(t0)) / max(int(s1) - int(s0), 1)
    mem = max(float(m[1]) for m in marks)
    print("%-10s bs=%-3d %6.3f s/iter peak %4.1fGB | 1.28M=%6.1fh 0.64M=%6.1fh 0.32M=%6.1fh"
          % (name, bs, per, mem,
             *(per * b / bs / 3600 for b in (1_280_000, 640_000, 320_000))))
else:
    print("%-10s bs=%-3d only %d marks parsed" % (name, bs, len(marks)))
PY
    break
  done
done
rm -rf experiments/timing_probe
echo PROBE_DONE
