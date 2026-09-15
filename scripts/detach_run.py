"""Detach a background Windows process so the agent shell cannot reap it."""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    script = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(script)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "batch.out")
    err_path = os.path.join(out_dir, "batch.err")
    # append mode so restarts keep history
    out = open(out_path, "a", encoding="utf-8", errors="replace", buffering=1)
    err = open(err_path, "a", encoding="utf-8", errors="replace", buffering=1)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    if script.lower().endswith(".bat") or script.lower().endswith(".cmd"):
        cmd = ["cmd.exe", "/c", script]
    elif script.lower().endswith(".ps1"):
        cmd = [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script,
        ]
    else:
        cmd = [sys.executable, script]
    p = subprocess.Popen(
        cmd,
        cwd=r"G:\RealSR",
        stdout=out,
        stderr=err,
        creationflags=flags,
        close_fds=True,
    )
    print(f"detached_pid={p.pid} cmd={cmd[0]} {script}")
    out.close()
    err.close()


if __name__ == "__main__":
    main()
