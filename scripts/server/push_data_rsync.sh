#!/usr/bin/env bash
# Resume RealSR tarball with rsync --partial
set -euo pipefail
LOCAL_TAR="/g/RealSR/RealSR(V3).tar.gz"
# WSL/git-bash path may differ; use scp from Windows side instead if needed
REMOTE=ds@10.222.120.101:/home/ds/realsr/data/
# prefer rsync if available
if command -v rsync >/dev/null 2>&1; then
  rsync -av --partial --progress "G:/RealSR/RealSR(V3).tar.gz" "$REMOTE"
else
  echo "no local rsync; use scp"
  scp "G:/RealSR/RealSR(V3).tar.gz" "$REMOTE"
fi
ssh "$REMOTE" true
ssh ds@10.222.120.101 "tar -xzf '/home/ds/realsr/data/RealSR(V3).tar.gz' -C /home/ds/realsr/data && ls /home/ds/realsr/data"
