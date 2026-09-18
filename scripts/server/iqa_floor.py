"""Perceptual floor: score the bicubic upsample itself on MUSIQ / MANIQA.

PSNR has an obvious floor (31.73 on these 100 pairs) but the two IQA metrics
have no reference point at all, so "MUSIQ 55.7" is meaningless on its own.
This scores the input everyone is trying to beat so the table has a bottom.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/ds/realsr")

from diffusion.data import build_index
from diffusion.iqa import IQAScorer
from diffusion.metrics import official_pair_metrics

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="data/RealSR(V3)")
p.add_argument("--scale", type=int, default=2)
p.add_argument("--out", default="experiments/diffusion/iqa_eval/bicubic_floor")
p.add_argument("--device", default="cuda")
args = p.parse_args()

pairs = build_index(args.data_root, ("Canon", "Nikon"), "Test", args.scale)
print(f"{len(pairs)} test pairs", flush=True)

scorer = IQAScorer(args.device)
scorer.available()

rows = []
for i, (lr_path, hr_path, sc) in enumerate(pairs):
    lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
    hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    sr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
    hh = min(sr.shape[0], hr.shape[0])
    ww = min(sr.shape[1], hr.shape[1])
    m = official_pair_metrics(sr[:hh, :ww], hr[:hh, :ww])
    m.update(scorer.score(sr[:hh, :ww]))
    m["name"] = Path(lr_path).name
    rows.append(m)
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(pairs)}] Y {m['psnr_y']:.2f}", flush=True)


def mean(k):
    vals = [r[k] for r in rows if k in r and r[k] == r[k]]
    return float(np.mean(vals)) if vals else float("nan")


summary = {
    "protocol": "bicubic upsample, RealSR official Test.m, limited-range Y",
    "n": len(rows),
    "psnr_y": mean("psnr_y"),
    "ssim_y": mean("ssim_y"),
    "psnr_rgb": mean("psnr_rgb"),
    "iqa_scores": {n: mean(n) for n in scorer.available()},
    "per_image": rows,
}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "eval.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))
