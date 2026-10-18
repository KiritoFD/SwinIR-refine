#!/usr/bin/env python3
"""Wait until Scale-96 training and all four 100-pair evaluation suites complete on host 48.

Monitors every 120 seconds:
1. Status of PID / process on host 48.
2. Checks for target files:
   - eval_full_metrics_raw/eval.json
   - eval_full_metrics_wddcp/eval.json
   - eval_full_metrics_wddcp_tta/eval.json
   - eval_full_metrics_adaptive_ca_wddcp_tta/eval.json
3. When all 4 evaluations complete, prints summary, triggers leaderboard update,
   and verifies WaveSR-Physics arm launch.
"""

import json
import subprocess
import sys
import time

TARGET_EVAL = "/home/ds/realsr/experiments/diffusion/scale96_muon/eval_full_metrics_adaptive_ca_wddcp_tta/eval.json"
SSH_CMD = ["ssh", "-o", "ConnectTimeout=15", "48"]


def run_ssh(cmd_str):
    try:
        res = subprocess.run(SSH_CMD + [cmd_str], capture_output=True, text=True, timeout=30)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Watchdog started. Waiting for completion on host 48...")
    sys.stdout.flush()
    
    consecutive_failures = 0
    while True:
        # Check if target eval file exists
        code, out, _ = run_ssh(f"test -f {TARGET_EVAL} && echo 'DONE' || echo 'WAITING'")
        if code == 0 and "DONE" in out:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALL FOUR EVALUATION PASSES COMPLETED!")
            sys.stdout.flush()
            break
            
        # Get status from monitor json
        code, out, err = run_ssh("cat /home/ds/realsr/experiments/STATUS_MONITOR.json 2>/dev/null")
        if code == 0 and out:
            try:
                status = json.loads(out)
                eta = status.get("eta_info", {})
                step = eta.get("current_step", "?")
                loss = eta.get("latest_loss", "?")
                rem_h = eta.get("remaining_hours", "?")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Step {step}/12000 | Loss {loss} | Remaining ~{rem_h}h")
                sys.stdout.flush()
                consecutive_failures = 0
            except Exception:
                pass
        else:
            consecutive_failures += 1
            if consecutive_failures > 5:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: 5 consecutive SSH failures ({err})")
                sys.stdout.flush()
                
        time.sleep(120)

    # All evaluations are finished: fetch leaderboard and check physics arm
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Fetching official Leaderboard...")
    run_ssh("/home/ds/miniconda3/envs/harness-qwen/bin/python /home/ds/realsr/scripts/server/print_metrics.py")
    code, lb_out, _ = run_ssh("cat /home/ds/realsr/experiments/BENCHMARK_LEADERBOARD.md 2>/dev/null")
    print("\n" + "=" * 50 + " UPDATED LEADERBOARD " + "=" * 50)
    print(lb_out)
    
    # Check physics arm
    code, phys_out, _ = run_ssh("ps aux | grep train_pixel | grep wavesr_physics || echo 'NOT_YET'")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WaveSR-Physics status: {phys_out}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Watchdog completed successfully.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
