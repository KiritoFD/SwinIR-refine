"""Speed/memory sweep over Mamba configs, to find what is actually trainable.

The question is not "is Mamba slow" (it is), but "which Mamba configurations can
reach a sample budget where a comparison with the U-Net means anything".  So for
each config: find the largest batch that fits (halve on OOM), run a few steps, and
report samples/s and tokens/s -- tokens/s is the architecture-level number, since
scale=2 works on 4096 tokens/sample and scale=1 on 16384.

An 8-step probe is ~1-2 min per config at these speeds, so this is cheap.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PY = "/home/ds/miniconda3/envs/harness-qwen/bin/python"
ROOT = "/home/ds/realsr"

# name, scale, window, dim, groups, blocks, expand, d_state, start_batch
CONFIGS = [
    ("s2_c128_base",   2,  0, 128, 4, 4, 2, 16, 320),
    ("s2_c128_st8",    2,  0, 128, 4, 4, 2,  8, 320),
    ("s2_c128_e1st8",  2,  0, 128, 4, 4, 1,  8, 512),
    ("s2_c96_light",   2,  0,  96, 2, 3, 1,  8, 512),
    ("s1w32_c128",     1, 32, 128, 4, 4, 2, 16,  64),
    ("s1w32_c128_st8", 1, 32, 128, 4, 4, 2,  8,  96),
    ("s1w16_c96",      1, 16,  96, 2, 3, 1,  8, 128),
    ("s1w32_c96_e1",   1, 32,  96, 2, 3, 1,  8, 128),
]

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
out = Path("/home/ds/realsr/experiments/diffusion/mamba_speed.json")
results = []

for name, scale, win, dim, groups, blocks, expand, dstate, batch0 in CONFIGS:
    entry = {"config": name, "scale": scale, "window": win, "dim": dim,
             "groups": groups, "blocks": blocks, "expand": expand, "d_state": dstate}
    batch = batch0
    ok = False
    while batch >= 8:
        cmd = [
            PY, "-u", "-m", "diffusion.train_pixel",
            "--data-root", "data/RealSR(V3)", "--out", "/tmp/mbprobe",
            "--backbone", "mamba", "--objective", "reg", "--residual", "1",
            "--base", str(dim), "--num-groups", str(groups), "--num-res", str(blocks),
            "--ssm-state", str(dstate), "--ssm-expand", str(expand),
            "--ssm-scale", str(scale), "--ssm-window", str(win),
            "--ssm-shift", "--ssm-backend", "mamba_ssm", "--grad-ckpt",
            "--lr-patch", "64", "--batch", str(batch), "--steps", str(STEPS),
            "--amp", "--num-workers", "8", "--cache-data", "0",
            "--decoded-manifest", "data/decoded/manifest.json", "--eval-every", "0",
        ]
        t0 = time.time()
        pr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            env={"HF_ENDPOINT": "https://hf-mirror.com",
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                 "PATH": "/usr/bin:/bin:/usr/local/bin"})
        wall = time.time() - t0
        txt = pr.stdout + pr.stderr
        if "OutOfMemoryError" in txt:
            print(f"{name:18s} batch {batch:4d} OOM", flush=True)
            batch //= 2
            continue
        steps = [l for l in txt.splitlines() if l.startswith("step ")]
        if len(steps) < 2:
            print(f"{name:18s} batch {batch:4d} FAILED: {txt.strip().splitlines()[-1][:80]}", flush=True)
            entry["error"] = txt.strip().splitlines()[-1][:200]
            break
        t_last = float(steps[-1].split("|")[-1].strip().rstrip("s"))
        t_prev = float(steps[-2].split("|")[-1].strip().rstrip("s"))
        s_per_step = (t_last - t_prev) / 50.0
        mem = steps[-1].split("|")[2].strip()
        toks = 16384 if scale == 1 else 4096
        entry.update({"batch": batch, "s_per_step": round(s_per_step, 3),
                      "samples_per_s": round(batch / s_per_step, 1),
                      "tokens_per_s": round(batch * toks / s_per_step),
                      "mem": mem,
                      "samples_in_1_5h": round(1.5 * 3600 / s_per_step * batch)})
        print(f"{name:18s} batch {batch:4d}  {s_per_step:6.3f} s/step  "
              f"{batch/s_per_step:7.1f} samp/s  {batch*toks/s_per_step/1000:7.0f}k tok/s  "
              f"{mem:>8s}  1.5h->{1.5*3600/s_per_step*batch:,.0f} samples", flush=True)
        ok = True
        break
    results.append(entry)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}")
