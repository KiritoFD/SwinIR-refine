"""Detach scp of RealSR tarball + extract on server."""
from __future__ import annotations

import os
import subprocess
import sys

LOCAL_TAR = r"G:\RealSR\RealSR(V3).tar.gz"
REMOTE = "ds@10.222.120.101"
REMOTE_DIR = "/home/ds/realsr/data"


def main() -> None:
    out_dir = r"G:\RealSR\experiments\diffusion\server_sync"
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "scp_data.log"), "a", encoding="utf-8", errors="replace")
    # 1) scp tarball
    cmd = [
        "scp", "-o", "BatchMode=yes",
        LOCAL_TAR,
        f"{REMOTE}:{REMOTE_DIR}/RealSR(V3).tar.gz",
    ]
    print("running", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print("scp rc", rc, flush=True)
    if rc != 0:
        sys.exit(rc)
    # 2) extract
    rcmd = (
        f"mkdir -p {REMOTE_DIR} && "
        f"tar -xzf '{REMOTE_DIR}/RealSR(V3).tar.gz' -C {REMOTE_DIR} && "
        f"ls -la {REMOTE_DIR} | head"
    )
    rc2 = subprocess.call(["ssh", REMOTE, rcmd], stdout=log, stderr=subprocess.STDOUT)
    print("extract rc", rc2, flush=True)
    log.close()


if __name__ == "__main__":
    main()
