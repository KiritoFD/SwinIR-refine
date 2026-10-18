"""Consolidate every eval.json found on the remote box into one CSV.

Source: .tmp_remote/experiments/**/eval.json (copied from ds@10.222.120.101:~/realsr).
Output: docs/results/all_evals.csv + a markdown summary printed to stdout.
"""
import csv
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".tmp_remote"))
OUT = os.path.join(os.path.dirname(__file__), "all_evals.csv")

COLS = [
    "experiment", "n", "psnr_y", "ssim_y", "psnr_rgb",
    "lpips", "dists", "niqe", "musiq", "maniqa",
    "steps", "tile", "pad", "mode", "objective",
]


def main():
    rows = []
    pattern = os.path.join(ROOT, "experiments", "**", "eval.json")
    for f in sorted(glob.glob(pattern, recursive=True)):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        iqa = d.get("iqa_scores") or {}
        rel = os.path.relpath(f, ROOT).replace("\\", "/")
        rel = rel.replace("experiments/diffusion/", "").replace("/eval.json", "")
        rows.append({
            "experiment": rel,
            "n": d.get("n"),
            "psnr_y": d.get("psnr_y"),
            "ssim_y": d.get("ssim_y"),
            "psnr_rgb": d.get("psnr_rgb"),
            "lpips": iqa.get("lpips"),
            "dists": iqa.get("dists"),
            "niqe": iqa.get("niqe"),
            "musiq": iqa.get("musiq"),
            "maniqa": iqa.get("maniqa"),
            "steps": d.get("steps"),
            "tile": d.get("tile"),
            "pad": d.get("pad"),
            "mode": d.get("mode"),
            "objective": d.get("objective"),
        })
    rows.sort(key=lambda r: -(r["psnr_y"] or 0))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")
    full = [r for r in rows if r["n"] == 100]
    print(f"of which full official 100-pair: {len(full)}")


if __name__ == "__main__":
    main()
