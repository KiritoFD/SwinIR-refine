#!/usr/bin/env python3
"""Watchdog and progress tracker for long-running experiments on host 48.

Monitors:
1. Scale-96 training progress, loss, lr, speed, and validation PSNR/SSIM jumps.
2. Checkpoint updates (ckpt_best.pt, ckpt_last.pt).
3. Post-training official 100-pair evaluations (Raw, WD-DCP, WD-DCP+TTA, Adaptive CA-WD-DCP+TTA).
4. Automatic Leaderboard regeneration via print_metrics.py.
5. WaveSR-Physics training progression.

Outputs a consolidated, real-time status summary to experiments/STATUS_MONITOR.json
and experiments/STATUS_MONITOR.md for instant query.
"""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/ds/realsr")
SCALE96_DIR = ROOT / "experiments/diffusion/scale96_muon"
PHYSICS_DIR = ROOT / "experiments/diffusion/wavesr_physics_b64"
STATUS_JSON = ROOT / "experiments/STATUS_MONITOR.json"
STATUS_MD = ROOT / "experiments/STATUS_MONITOR.md"


def get_latest_train_log():
    train_log_file = SCALE96_DIR / "train_log.jsonl"
    if not train_log_file.exists():
        return None
    try:
        lines = train_log_file.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            return json.loads(lines[-1])
    except Exception:
        pass
    return None


def get_val_history():
    val_log_file = SCALE96_DIR / "val_log.jsonl"
    if not val_log_file.exists():
        return []
    res = []
    try:
        for line in val_log_file.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                res.append(json.loads(line))
    except Exception:
        pass
    return res


def check_eval_results(exp_dir):
    evals = {}
    candidates = {
        "raw": exp_dir / "eval_full_metrics_raw/eval.json",
        "wddcp": exp_dir / "eval_full_metrics_wddcp/eval.json",
        "wddcp_tta": exp_dir / "eval_full_metrics_wddcp_tta/eval.json",
        "adaptive_ca_wddcp_tta": exp_dir / "eval_full_metrics_adaptive_ca_wddcp_tta/eval.json",
    }
    for k, p in candidates.items():
        if p.exists():
            try:
                evals[k] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return evals


def update_leaderboard():
    script = ROOT / "scripts/server/print_metrics.py"
    if script.exists():
        try:
            subprocess.run(["/home/ds/miniconda3/envs/harness-qwen/bin/python", str(script)],
                           cwd=str(ROOT), capture_output=True, timeout=60)
        except Exception:
            pass


def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monitor daemon started.")
    last_val_count = 0
    leaderboard_updated = False
    
    while True:
        try:
            latest = get_latest_train_log()
            vals = get_val_history()
            scale96_evals = check_eval_results(SCALE96_DIR)
            
            # Check if new validation appeared
            if len(vals) > last_val_count:
                last_val = vals[-1]
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New validation: Step {last_val.get('step')}: "
                      f"PSNR-Y {last_val.get('psnr_y'):.4f}, SSIM-Y {last_val.get('ssim_y'):.4f}")
                last_val_count = len(vals)
                
            # If any evaluations exist and leaderboard not yet updated
            if scale96_evals and not leaderboard_updated:
                update_leaderboard()
                leaderboard_updated = True
                
            # Compute remaining time if training
            eta_info = {}
            if latest:
                cur_step = latest.get("step", 0)
                total_steps = 12000
                if cur_step > 0:
                    # check elapsed from train.log
                    train_log = SCALE96_DIR / "train.log"
                    elapsed_s = 0
                    if train_log.exists():
                        for line in reversed(train_log.read_text(encoding="utf-8").splitlines()[-10:]):
                            if f"step {cur_step:06d}" in line:
                                parts = line.split("|")
                                for p in parts:
                                    if "s" in p and p.strip().replace("s", "").isdigit():
                                        elapsed_s = int(p.strip().replace("s", ""))
                                        break
                                break
                    if elapsed_s > 0 and cur_step > 0:
                        speed = elapsed_s / cur_step
                        rem_steps = max(0, total_steps - cur_step)
                        rem_s = rem_steps * speed
                        eta_info = {
                            "current_step": cur_step,
                            "total_steps": total_steps,
                            "elapsed_seconds": elapsed_s,
                            "speed_s_per_step": round(speed, 3),
                            "remaining_seconds": round(rem_s),
                            "remaining_hours": round(rem_s / 3600.0, 2),
                            "progress_pct": round(cur_step / total_steps * 100, 2),
                            "latest_loss": latest.get("loss"),
                            "latest_lr": latest.get("lr"),
                        }
                        
            # Dump JSON state
            state = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "eta_info": eta_info,
                "validation_history": vals,
                "scale96_evaluations": {k: {"psnr_y": v.get("psnr_y"), "ssim_y": v.get("ssim_y")} for k, v in scale96_evals.items()},
                "physics_arm_started": PHYSICS_DIR.exists() and (PHYSICS_DIR / "train.log").exists(),
            }
            STATUS_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
            
            # Dump readable Markdown state
            md = [
                f"# RealSR Experiment Monitor ({state['timestamp']})\n",
                f"- **Progress**: Step {eta_info.get('current_step', 0)} / {eta_info.get('total_steps', 12000)} ({eta_info.get('progress_pct', 0)}%)",
                f"- **Loss**: {eta_info.get('latest_loss', '-')} | **LR**: {eta_info.get('latest_lr', '-')}",
                f"- **Speed**: {eta_info.get('speed_s_per_step', '-')} s/step | **Remaining**: ~{eta_info.get('remaining_hours', '-')} hours",
                "\n### Validation History (16-pair Test Crops)",
            ]
            for v in vals:
                md.append(f"- Step {v.get('step')}: **PSNR-Y {v.get('psnr_y'):.4f} dB** | SSIM-Y {v.get('ssim_y'):.4f}")
            if scale96_evals:
                md.append("\n### Scale-96 Full 100-Pair Official Evaluations")
                for k, v in scale96_evals.items():
                    md.append(f"- **{k}**: PSNR-Y {v.get('psnr_y'):.4f} dB | SSIM-Y {v.get('ssim_y'):.4f}")
            if state["physics_arm_started"]:
                md.append("\n- [INFO] WaveSR-Physics training has started!")
            STATUS_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
            
        except Exception as e:
            pass
            
        time.sleep(30)


if __name__ == "__main__":
    main()
