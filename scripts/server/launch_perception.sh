#!/usr/bin/env bash
cd /home/ds/realsr || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
bash -n scripts/server/run_perception.sh && echo SH_OK
tmux kill-session -t perc 2>/dev/null
tmux new-session -d -s perc 'bash scripts/server/run_perception.sh > experiments/perception.log 2>&1'
sleep 4; echo "--- sessions ---"; tmux ls; echo "--- log ---"; cat experiments/perception.log
