"""Serial official RealSR evals, one model at a time, VRAM-aware (tile=96)."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, r"G:\RealSR")

ROOT = Path(r"G:\RealSR\experiments\eval_official\batch_20260915")
DATA = r"G:\RealSR\data\RealSR(V3)"
PY = r"C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"

JOBS = [
    ("E10_realsr_loss", r"G:\RealSR\experiments\improve\E10_realsr_loss_15k\ckpt_best.pt"),
    ("A1_l1_ema", r"G:\RealSR\experiments\ablation_x2\A1_l1_ema\ckpt_best.pt"),
    ("A2_l1_ema_patchamp", r"G:\RealSR\experiments\ablation_x2\A2_l1_ema_patchamp\ckpt_best.pt"),
    ("A3_l1_ema_hfconf", r"G:\RealSR\experiments\ablation_x2\A3_l1_ema_hfconf\ckpt_best.pt"),
    ("A4_full", r"G:\RealSR\experiments\ablation_x2\A4_full\ckpt_best.pt"),
    ("E7_v2_l1_amp_ema", r"G:\RealSR\experiments\improve\E7_mod_v2_l1_amp_ema\ckpt_best.pt"),
    ("E8_v2_kpn", r"G:\RealSR\experiments\improve\E8_v2_kpn\ckpt_best.pt"),
    ("E12_lpkpn", r"G:\RealSR\experiments\improve\E12_align_lpkpn\ckpt_best.pt"),
    ("E13_wiener", r"G:\RealSR\experiments\improve\E13_align_wiener\ckpt_best.pt"),
    ("E14_ampphase", r"G:\RealSR\experiments\improve\E14_align_ampphase\ckpt_best.pt"),
    ("E15_radialpsf", r"G:\RealSR\experiments\improve\E15_align_radialpsf\ckpt_best.pt"),
]


def reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def run_one(name: str, ckpt: str) -> dict:
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    # isolate eval in subprocess so a CUDA error cannot poison the next model
    cmd = (
        f'"{PY}" -m model.eval --ckpt "{ckpt}" --data-root "{DATA}" '
        f"--scale 2 --max-pairs 0 --out \"{out}\" --tile 96 --resume"
    )
    print(f"==== {name} ====", flush=True)
    t0 = time.time()
    # run via cmd so env is clean; capture combined log
    log_path = out / "eval_run.log"
    rc = os.system(f'cmd /c "{cmd} > "{log_path}" 2>&1"')
    dt = time.time() - t0
    ej = out / "eval.json"
    if ej.exists():
        data = json.loads(ej.read_text(encoding="utf-8"))
        row = {
            "name": name,
            "n": data.get("n"),
            "Y": data.get("psnr_y"),
            "SSIM_Y": data.get("ssim_y"),
            "RGB": data.get("psnr_rgb"),
            "sec": round(dt, 1),
            "ok": True,
        }
        print(
            f"  OK Y={row['Y']:.4f} SSIM={row['SSIM_Y']:.4f} RGB={row['RGB']:.4f} "
            f"n={row['n']} {row['sec']}s",
            flush=True,
        )
    else:
        row = {"name": name, "n": 0, "Y": None, "SSIM_Y": None, "RGB": None,
               "sec": round(dt, 1), "ok": False, "rc": rc}
        print(f"  FAIL rc={rc} {dt:.1f}s see {log_path}", flush=True)
        # print last lines of log
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in lines[-12:]:
                print("  |", ln, flush=True)
        except OSError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "summary.json"
    existing: dict[str, dict] = {}
    if summary_path.exists():
        try:
            for r in json.loads(summary_path.read_text(encoding="utf-8")):
                if r.get("ok") and r.get("n"):
                    existing[r["name"]] = r
        except Exception:
            pass
    # also accept any eval.json already on disk
    for name, _ckpt in JOBS:
        ej = ROOT / name / "eval.json"
        if ej.exists() and name not in existing:
            try:
                data = json.loads(ej.read_text(encoding="utf-8"))
                if data.get("n"):
                    existing[name] = {
                        "name": name,
                        "n": data.get("n"),
                        "Y": data.get("psnr_y"),
                        "SSIM_Y": data.get("ssim_y"),
                        "RGB": data.get("psnr_rgb"),
                        "ok": True,
                    }
                    print(f"SKIP {name} (eval.json exists)", flush=True)
            except Exception:
                pass

    rows: list[dict] = []
    for name, ckpt in JOBS:
        if name in existing:
            rows.append(existing[name])
            _write_summary(summary_path, rows)
            continue
        if not Path(ckpt).exists():
            print(f"MISSING ckpt {name}: {ckpt}", flush=True)
            rows.append({"name": name, "n": 0, "Y": None, "SSIM_Y": None,
                         "RGB": None, "ok": False, "error": "no ckpt"})
            _write_summary(summary_path, rows)
            continue
        reset_cuda()
        try:
            row = run_one(name, ckpt)
        except Exception:
            traceback.print_exc()
            row = {"name": name, "n": 0, "Y": None, "SSIM_Y": None,
                   "RGB": None, "ok": False, "error": traceback.format_exc()[-300:]}
        rows.append(row)
        _write_summary(summary_path, rows)

    print("\n==== SUMMARY ====", flush=True)
    for r in rows:
        if r.get("Y") is not None:
            print(f"{r['name']:22s} n={r['n']:3d}  Y={r['Y']:.4f}  SSIM={r['SSIM_Y']:.4f}  RGB={r['RGB']:.4f}")
        else:
            print(f"{r['name']:22s} FAILED")
    print("DONE", flush=True)


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
