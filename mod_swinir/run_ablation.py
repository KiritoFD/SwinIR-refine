"""Stepwise loss-component ablation on RealSR V3 ×2 (fixed arch/recipe)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "experiments" / "ablation_x2"

# Fixed protocol — only loss flags change.
# Model: Mod v2 base 4.15M (residual bicubic, AMF-v2, FiLM-RAPE, DCN, KPN gate)
# Train: batch=8, LR64, steps=6000, AMP fp16, cosine, warmup 150
# Eval: 6 tiled Test pairs every 1000 steps

ABLATIONS = [
    dict(
        id="A0_l1_only",
        w_amp=0.0,
        w_hf_conf=0.0,
        ema=0.0,
        note="L1 only, no EMA",
    ),
    dict(
        id="A1_l1_ema",
        w_amp=0.0,
        w_hf_conf=0.0,
        ema=0.999,
        note="+ EMA",
    ),
    dict(
        id="A2_l1_ema_patchamp",
        w_amp=0.05,
        w_hf_conf=0.0,
        ema=0.999,
        note="+ patch amplitude spectrum",
    ),
    dict(
        id="A3_l1_ema_hfconf",
        w_amp=0.0,
        w_hf_conf=0.25,
        ema=0.999,
        note="+ confidence-weighted HF",
    ),
    dict(
        id="A4_full",
        w_amp=0.05,
        w_hf_conf=0.25,
        ema=0.999,
        note="L1 + patch-amp + conf-HF + EMA (E10 recipe)",
    ),
]


def run_one(exp):
    out = OUT / exp["id"]
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    if summary.exists():
        print(f"[skip] {exp['id']}")
        return json.loads(summary.read_text(encoding="utf-8"))

    cmd = [
        PY,
        "-m",
        "mod_swinir.train",
        "--data-root",
        r"G:\RealSR\data\RealSR(V3)",
        "--out",
        str(out),
        "--scale",
        "2",
        "--arch",
        "mod",
        "--model-size",
        "base",
        "--batch-size",
        "8",
        "--grad-accum",
        "1",
        "--lr-patch",
        "64",
        "--steps",
        "6000",
        "--lr",
        "2e-4",
        "--warmup",
        "150",
        "--num-workers",
        "0",
        "--eval-every",
        "1000",
        "--eval-pairs",
        "6",
        "--save-every",
        "2000",
        "--cameras",
        "Canon,Nikon",
        "--amp",
        "--amp-dtype",
        "fp16",
        "--w-amp",
        str(exp["w_amp"]),
        "--w-hf-conf",
        str(exp["w_hf_conf"]),
    ]
    if exp["ema"] > 0:
        cmd += ["--ema", str(exp["ema"])]

    print(f"\n===== {exp['id']}: {exp['note']} =====", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[fail] {exp['id']}")
        return {"id": exp["id"], "error": r.returncode}

    evals = []
    el = out / "eval_log.jsonl"
    if el.exists():
        for line in el.read_text(encoding="utf-8").splitlines():
            if line.strip():
                evals.append(json.loads(line))
    best = max((e["psnr"] for e in evals), default=-1)
    best_s = max((e["ssim"] for e in evals if e.get("psnr") == best), default=0)
    rec = {
        "id": exp["id"],
        "note": exp["note"],
        "w_amp": exp["w_amp"],
        "w_hf_conf": exp["w_hf_conf"],
        "ema": exp["ema"],
        "best_psnr": best,
        "best_ssim": best_s,
        "wall_sec": time.time() - t0,
        "evals": evals,
    }
    summary.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[done] {exp['id']} best={best:.2f} dB")
    return rec


def write_report(results):
    lines = [
        "# RealSR V3 ×2 — Loss Component Ablation",
        "",
        "Fixed: ModSwinIR-v2 base 4.15M, residual bicubic, batch=8, LR64, 6k steps, AMP fp16.",
        "Eval: 6 tiled Test pairs every 1k steps (online protocol).",
        "",
        "| ID | L1 | patch-amp | conf-HF | EMA | Best PSNR | Best SSIM | Note |",
        "|----|----|-----------|---------|-----|-----------|-----------|------|",
    ]
    for r in results:
        if "best_psnr" not in r:
            lines.append(f"| {r.get('id')} | | | | | FAIL | | {r.get('error','')} |")
            continue
        lines.append(
            f"| {r['id']} | ✓ | {r['w_amp'] or '—'} | {r['w_hf_conf'] or '—'} | "
            f"{'✓' if r['ema'] else '—'} | **{r['best_psnr']:.2f}** | {r['best_ssim']:.4f} | {r['note']} |"
        )
    lines.append("")
    lines.append("Delta vs A1 (L1+EMA):")
    base = next((r for r in results if r.get("id") == "A1_l1_ema" and "best_psnr" in r), None)
    if base:
        for r in results:
            if "best_psnr" in r and r["id"] != base["id"]:
                lines.append(f"- {r['id']}: {r['best_psnr'] - base['best_psnr']:+.2f} dB")
    else:
        a0 = next((r for r in results if r.get("id") == "A0_l1_only" and "best_psnr" in r), None)
        if a0:
            lines.append("(vs A0 L1-only):")
            for r in results:
                if "best_psnr" in r and r["id"] != a0["id"]:
                    lines.append(f"- {r['id']}: {r['best_psnr'] - a0['best_psnr']:+.2f} dB")
    path = OUT / "ABLATION_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    for exp in ABLATIONS:
        if only and exp["id"] not in only:
            continue
        if time.time() - t0 > 11 * 3600:
            print("budget stop")
            break
        results.append(run_one(exp))
        write_report(results)
    write_report(results)
    print(f"ablation wall {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
