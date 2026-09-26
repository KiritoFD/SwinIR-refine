#!/usr/bin/env bash
# Launcher: start the baseline-reproduction chain in tmux (keeps the ssh command
# free of nested quotes/redirection that PowerShell mangles).
cd /home/ds/realsr || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
tmux kill-session -t baserep 2>/dev/null
tmux new-session -d -s baserep 'bash scripts/server/run_baselines.sh > experiments/baselines.log 2>&1'
sleep 5
echo "--- sessions ---"; tmux ls
echo "--- baselines.log ---"; cat experiments/baselines.log 2>/dev/null
echo "--- edsr train head ---"; head -6 experiments/diffusion/logs/bl_edsr.train.log 2>/dev/null
echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
