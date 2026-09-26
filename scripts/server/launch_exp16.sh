#!/usr/bin/env bash
cd /home/ds/realsr || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
bash -n scripts/server/run_exp16.sh && echo SH_OK
tmux kill-session -t exp16 2>/dev/null
tmux new-session -d -s exp16 'bash scripts/server/run_exp16.sh > experiments/exp16.log 2>&1'
sleep 6; echo "--- sessions ---"; tmux ls; echo "--- exp16.log ---"; cat experiments/exp16.log; echo "--- first arm head ---"; head -6 experiments/diffusion/logs/ex_ot_ot_swd_img_add.train.log 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
