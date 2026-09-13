"""Run experiment matrix on RealSR V3 x2 within a time budget."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "experiments" / "matrix_x2"

# Fair protocol: same data, same steps-to-samples, periodic eval.
# Effective samples ≈ steps * batch * accum. Aim ~32k samples like prior base run.
#
# | id | arch | size | params | batch | accum | patch | steps | notes |
# E1 SwinIR-light official under-capacity baseline
# E2 SwinIR-mid embed96 capacity-matched to Mod-base
# E3 ModSwinIR-base (already trained — optional rerun)
# E4 ModSwinIR-large fixed (batch=1, bf16, lower lr)
# E5 SwinIR-classical-ish largeish 10M (batch=1) if time

# Fair protocol: same data, similar effective samples, periodic eval.
# Actual params: SwinIR-light 0.61M, largeish 3.96M, classical 11.75M; Mod base 4.12M, large 10.21M

EXPERIMENTS = [
    dict(
        id="E1_swinir_light",
        arch="swinir",
        model_size="light",
        batch_size=8,
        grad_accum=1,
        lr_patch=64,
        steps=4000,
        lr=2e-4,
        amp=False,
        amp_dtype="fp16",
        eval_every=500,
        save_every=1000,
        note="SwinIR-light official 0.61M, L1, under-capacity baseline",
    ),
    dict(
        id="E2_swinir_capmatch",
        arch="swinir",
        model_size="largeish",
        batch_size=4,
        grad_accum=2,
        lr_patch=64,
        steps=4000,
        lr=2e-4,
        amp=True,
        amp_dtype="fp16",
        eval_every=500,
        save_every=1000,
        note="SwinIR-largeish 3.96M, L1, capacity-matched to Mod-base",
    ),
    dict(
        id="E3_mod_base",
        arch="mod",
        model_size="base",
        batch_size=8,
        grad_accum=1,
        lr_patch=64,
        steps=4000,
        lr=2e-4,
        amp=True,
        amp_dtype="fp16",
        eval_every=500,
        save_every=1000,
        note="ModSwinIR-base 4.12M UWCL (reuse prior run)",
        reuse_dir=ROOT / "experiments" / "mod_swinir_x2_base",
    ),
    dict(
        id="E4_mod_large",
        arch="mod",
        model_size="large",
        batch_size=1,
        grad_accum=8,
        lr_patch=96,
        steps=6000,
        lr=1e-4,
        amp=True,
        amp_dtype="bf16",
        eval_every=750,
        save_every=1500,
        note="ModSwinIR-large 10.21M, batch=1, bf16, lr=1e-4 (NaN-hardened)",
    ),
    dict(
        id="E5_swinir_classical",
        arch="swinir",
        model_size="classical",
        batch_size=1,
        grad_accum=8,
        lr_patch=80,
        steps=4000,
        lr=1e-4,
        amp=True,
        amp_dtype="bf16",
        eval_every=500,
        save_every=1000,
        note="SwinIR-classical 11.75M official, L1, batch=1 full-capacity baseline",
    ),
]


def run_one(exp, skip_if_done=True):
    out = OUT / exp["id"]
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    if skip_if_done and summary.exists():
        print(f"[skip] {exp['id']} already has summary.json")
        return json.loads(summary.read_text(encoding="utf-8"))

    reuse = exp.get("reuse_dir")
    if reuse and (reuse / "eval_log.jsonl").exists() and skip_if_done:
        # copy/link prior result into matrix
        prior = json.loads((reuse / "summary.json").read_text(encoding="utf-8")) if (reuse / "summary.json").exists() else {}
        evals = []
        el = reuse / "eval_log.jsonl"
        if el.exists():
            for line in el.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    evals.append(json.loads(line))
        best = prior.get("best_psnr") or (max((e["psnr"] for e in evals), default=-1))
        rec = {
            "id": exp["id"],
            "reused_from": str(reuse),
            "params_m": prior.get("params_m") or prior.get("args", {}).get("model_size"),
            "best_psnr": best,
            "evals": evals,
            "note": exp["note"],
        }
        summary.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[reuse] {exp['id']} best_psnr={best}")
        return rec

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
        exp["arch"],
        "--model-size",
        exp["model_size"],
        "--batch-size",
        str(exp["batch_size"]),
        "--grad-accum",
        str(exp["grad_accum"]),
        "--lr-patch",
        str(exp["lr_patch"]),
        "--steps",
        str(exp["steps"]),
        "--lr",
        str(exp["lr"]),
        "--warmup",
        "100",
        "--num-workers",
        "4",
        "--eval-every",
        str(exp["eval_every"]),
        "--eval-pairs",
        "4",
        "--save-every",
        str(exp["save_every"]),
        "--cameras",
        "Canon,Nikon",
    ]
    if exp.get("amp"):
        cmd += ["--amp", "--amp-dtype", exp.get("amp_dtype", "fp16")]
    if exp["arch"] == "swinir":
        # default L1-only; set --use-uwcl to force UWCL
        pass

    print(f"\n===== RUN {exp['id']}: {exp['note']} =====")
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[fail] {exp['id']} rc={proc.returncode}")
        return {"id": exp["id"], "error": proc.returncode}

    evals = []
    el = out / "eval_log.jsonl"
    if el.exists():
        for line in el.read_text(encoding="utf-8").splitlines():
            if line.strip():
                evals.append(json.loads(line))
    best = max((e["psnr"] for e in evals), default=-1)
    rec = {
        "id": exp["id"],
        "arch": exp["arch"],
        "model_size": exp["model_size"],
        "params_m": None,
        "best_psnr": best,
        "wall_sec": dt,
        "evals": evals,
        "note": exp["note"],
        "cmd": cmd,
    }
    sm = out / "summary.json"
    if sm.exists():
        rec.update(json.loads(sm.read_text(encoding="utf-8")))
        rec["id"] = exp["id"]
        rec["best_psnr"] = max(best, rec.get("best_psnr") or -1)
    sm.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[done] {exp['id']} best={rec['best_psnr']:.2f} dB in {dt/60:.1f} min")
    return rec


def write_report(results):
    lines = ["# RealSR V3 x2 Experiment Matrix", ""]
    lines.append("| ID | Arch | Size | Params | Best PSNR | Best SSIM | Note |")
    lines.append("|----|------|------|--------|-----------|-----------|------|")
    for r in results:
        if not r or "error" in r and "best_psnr" not in r:
            continue
        evals = r.get("evals") or []
        best_p = r.get("best_psnr", -1)
        best_s = 0
        for e in evals:
            if e.get("psnr", -1) == best_p:
                best_s = e.get("ssim", 0)
                break
        if not best_s and evals:
            best_s = max(e.get("ssim", 0) for e in evals)
        lines.append(
            f"| {r.get('id')} | {r.get('arch', r.get('reused_from', '')[:20])} | "
            f"{r.get('model_size', '?')} | {r.get('params_m') or '?'} | "
            f"{best_p:.2f} | {best_s:.4f} | {r.get('note', '')} |"
        )
    lines.append("")
    lines.append("Protocol: Canon+Nikon Train scale2, Test tiled eval 4 pairs every N steps, LR cosine.")
    path = OUT / "MATRIX_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "matrix_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    budget = 12 * 3600
    for exp in EXPERIMENTS:
        if only and exp["id"] not in only and not any(exp["id"].startswith(o) for o in only):
            continue
        if time.time() - t0 > budget * 0.92:
            print("[budget] stop early")
            break
        results.append(run_one(exp))
        write_report(results)
    write_report(results)
    print(f"matrix wall time {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
