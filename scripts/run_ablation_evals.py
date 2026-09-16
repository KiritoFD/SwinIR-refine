"""Official full-set eval for every run in ablation_matrix.json.

Serial, one model at a time, tile=96, --resume (per_image.jsonl).
Skips runs whose eval.json already has n>=100 (or n>0 if allow_partial).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\RealSR")
MATRIX = ROOT / "scripts" / "ablation_matrix.json"
OUT_ROOT = ROOT / "experiments" / "eval_official" / "ablation_full_20260915"
PY = r"C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
DATA = r"G:\RealSR\data\RealSR(V3)"


def load_matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def eval_out_dir(run_id: str) -> Path:
    return OUT_ROOT / run_id


def has_full_eval(out_dir: Path, min_n: int = 100) -> bool:
    ej = out_dir / "eval.json"
    if not ej.exists():
        return False
    try:
        data = json.loads(ej.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(data.get("n") or 0) >= min_n


# reuse official evals already produced in earlier batches
LEGACY_EVAL = {
    "E11_align": "batch_20260914/E11_align_best",
    "E9_no_align": "batch_20260914/E9_long12k_best",
    "A0_no_ema": "batch_20260914/A0_l1_only_best",
    "A1_ema": "batch_20260915/A1_l1_ema",
    "A2_patchamp": "batch_20260915/A2_l1_ema_patchamp",
    "A3_hfconf": "batch_20260915/A3_l1_ema_hfconf",
    "A4_full_aux": "batch_20260915/A4_full",
    "E10_realsr_loss": "batch_20260915/E10_realsr_loss",
    "E12_lpkpn": "batch_20260915/E12_lpkpn",
    "E13_wiener": "batch_20260915/E13_wiener",
    "E14_ampphase": "batch_20260915/E14_ampphase",
    "E15_radialpsf": "batch_20260915/E15_radialpsf",
    "E1_swinir_light": "batch_20260914/E1_swinir_light",
    "E2_swinir_capmatch": "batch_20260914/E2_swinir_capmatch",
}


def seed_from_legacy(run_id: str, out_dir: Path) -> bool:
    """Copy a completed official eval.json from an earlier batch if present."""
    if has_full_eval(out_dir):
        return True
    rel = LEGACY_EVAL.get(run_id)
    if not rel:
        return False
    src = ROOT / "experiments" / "eval_official" / rel / "eval.json"
    if not src.exists():
        return False
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(data.get("n") or 0) < 100:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"SEED {run_id} from {rel}", flush=True)
    return True


def run_one(run: dict, min_n: int) -> dict:
    rid = run["id"]
    ckpt = ROOT / run["ckpt"]
    out = eval_out_dir(rid)
    out.mkdir(parents=True, exist_ok=True)
    row = {
        "id": rid,
        "role": run.get("role"),
        "factor": run.get("factor"),
        "ckpt": str(ckpt),
        "note": run.get("note", ""),
    }
    if run.get("requires_train") and not ckpt.exists():
        row.update({"ok": False, "error": "ckpt missing (train first)", "n": 0,
                    "Y": None, "SSIM_Y": None, "RGB": None})
        print(f"SKIP-TRAIN {rid}: no ckpt", flush=True)
        return row
    if not ckpt.exists():
        row.update({"ok": False, "error": "ckpt missing", "n": 0,
                    "Y": None, "SSIM_Y": None, "RGB": None})
        print(f"FAIL {rid}: ckpt not found {ckpt}", flush=True)
        return row
    seed_from_legacy(rid, out)
    if has_full_eval(out, min_n=min_n):
        data = json.loads((out / "eval.json").read_text(encoding="utf-8"))
        row.update({
            "ok": True, "n": data.get("n"),
            "Y": data.get("psnr_y"), "SSIM_Y": data.get("ssim_y"),
            "RGB": data.get("psnr_rgb"), "cached": True,
        })
        print(f"CACHED {rid} n={row['n']} Y={row['Y']:.4f}", flush=True)
        return row

    print(f"==== {rid} ====", flush=True)
    t0 = time.time()
    log_path = out / "eval_run.log"
    cmd = [
        PY, "-m", "model.eval",
        "--ckpt", str(ckpt),
        "--data-root", DATA,
        "--scale", "2",
        "--max-pairs", "0",
        "--out", str(out),
        "--tile", "96",
        "--resume",
    ]
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("a", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
            env=env, timeout=3600,
        )
    dt = time.time() - t0
    ej = out / "eval.json"
    if ej.exists():
        data = json.loads(ej.read_text(encoding="utf-8"))
        row.update({
            "ok": True, "n": data.get("n"),
            "Y": data.get("psnr_y"), "SSIM_Y": data.get("ssim_y"),
            "RGB": data.get("psnr_rgb"),
            "sec": round(dt, 1), "rc": proc.returncode,
        })
        print(f"  OK n={row['n']} Y={row['Y']:.4f} SSIM={row['SSIM_Y']:.4f} "
              f"RGB={row['RGB']:.4f} {row['sec']}s", flush=True)
    else:
        # recover partial mean from per_image.jsonl
        partial = out / "per_image.jsonl"
        n_ok, ys = 0, []
        if partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in rec:
                    continue
                y = rec.get("psnr_y")
                if y is not None and y == y:  # not NaN
                    n_ok += 1
                    ys.append(float(y))
        row.update({
            "ok": False, "n": n_ok,
            "Y": (sum(ys) / len(ys)) if ys else None,
            "SSIM_Y": None, "RGB": None,
            "sec": round(dt, 1), "rc": proc.returncode,
            "error": f"eval.json missing rc={proc.returncode}",
        })
        print(f"  FAIL rc={proc.returncode} partial_n={n_ok} {dt:.1f}s", flush=True)
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in lines[-8:]:
                print("  |", ln, flush=True)
        except OSError:
            pass
    gc.collect()
    return row


def write_summary(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # markdown table
    lines = [
        "# Ablation full-set official results",
        "",
        "Protocol: RealSR V3 Test.m, n=100 (Y limited-range uint8, crop=0, modcrop=4)",
        "",
        "| id | role | factor | n | Y-PSNR | Y-SSIM | RGB | note |",
        "|----|------|--------|---|--------|--------|-----|------|",
    ]
    for r in rows:
        y = f"{r['Y']:.4f}" if r.get("Y") is not None else "—"
        s = f"{r['SSIM_Y']:.4f}" if r.get("SSIM_Y") is not None else "—"
        rgb = f"{r['RGB']:.4f}" if r.get("RGB") is not None else "—"
        n = r.get("n", 0)
        note = (r.get("note") or "").replace("|", "/")
        lines.append(
            f"| {r['id']} | {r.get('role','')} | {r.get('factor','')} | "
            f"{n} | {y} | {s} | {rgb} | {note} |"
        )
    lines.append("")
    # comparisons vs baseline if present
    by_id = {r["id"]: r for r in rows}
    base = by_id.get("E11_align")
    if base and base.get("Y") is not None:
        lines += ["## Δ vs E11_align (Y)", ""]
        lines.append("| id | Y | Δ Y |")
        lines.append("|----|---|-----|")
        for r in rows:
            if r["id"] == "E11_align" or r.get("Y") is None:
                continue
            lines.append(f"| {r['id']} | {r['Y']:.4f} | {r['Y']-base['Y']:+.4f} |")
        lines.append("")
    (OUT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_ROOT/'summary.json'} and RESULTS.md", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--only", type=str, default="", help="comma-separated run ids")
    args = ap.parse_args()
    matrix = load_matrix()
    runs = matrix["runs"]
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        runs = [r for r in runs if r["id"] in keep]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run in runs:
        try:
            row = run_one(run, min_n=args.min_n)
        except Exception as e:  # noqa: BLE001
            row = {
                "id": run["id"], "ok": False, "n": 0, "Y": None,
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"EXC {run['id']}: {e}", flush=True)
        rows.append(row)
        write_summary(rows)
    write_summary(rows)
    print("DONE ablation full evals", flush=True)


if __name__ == "__main__":
    main()
