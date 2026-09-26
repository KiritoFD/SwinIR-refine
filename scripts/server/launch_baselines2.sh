#!/usr/bin/env bash
# Launcher for the second baseline batch (SRResNet, RRDB) in its own tmux session.
cd /home/ds/realsr || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
tmux kill-session -t baserep2 2>/dev/null
tmux new-session -d -s baserep2 'bash scripts/server/run_baselines2.sh > experiments/baselines2.log 2>&1'
sleep 5
echo "--- sessions ---"; tmux ls
echo "--- baselines2.log ---"; cat experiments/baselines2.log 2>/dev/null
echo "--- baselines.log ---"; tail -12 experiments/baselines.log 2>/dev/null
