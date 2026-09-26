#!/usr/bin/env bash
cd /home/ds/realsr || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
bash -n scripts/server/run_ot.sh && echo SH_OK
tmux kill-session -t ot 2>/dev/null
tmux new-session -d -s ot 'bash scripts/server/run_ot.sh > experiments/ot.log 2>&1'
sleep 3; echo "--- sessions ---"; tmux ls; echo "--- ot.log ---"; cat experiments/ot.log
