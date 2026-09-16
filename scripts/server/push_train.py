"""Upload RealSR Train folders via scp (Canon then Nikon)."""
from __future__ import annotations

import os
import subprocess
import time

REMOTE = "ds@10.222.120.101"
BASE = "/home/ds/realsr/data/RealSR(V3)"
LOG = r"G:\RealSR\experiments\diffusion\server_sync\scp_train.log"


def log(msg: str) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    for cam in ("Canon", "Nikon"):
        src = rf"G:\RealSR\data\RealSR(V3)\{cam}\Train"
        dst = f"{REMOTE}:{BASE}/{cam}/"
        log(f"scp {cam} Train -> {dst}")
        rc = subprocess.call(["scp", "-o", "BatchMode=yes", "-r", src, dst])
        log(f"scp {cam} rc={rc}")
    rc = subprocess.call(
        ["ssh", REMOTE, f"find {BASE} -name '*_LR2.png' | wc -l; du -sh {BASE}"]
    )
    log(f"count rc={rc} ALL_DONE")


if __name__ == "__main__":
    main()
