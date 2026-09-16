#!/usr/bin/env bash
# Launch the 36h matrix inside a tmux session (detached, survives ssh drops).
#
#   bash scripts/server/start_36h.sh      # start / restart
#   bash scripts/server/status_36h.sh     # progress
#   tmux attach -t realsr36h              # watch live
set -uo pipefail

SESS="${SESS:-realsr36h}"
ROOT="${ROOT:-/home/ds/realsr}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found; running in foreground with nohup instead"
  cd "$ROOT" && nohup bash scripts/server/${RUN:-run_36h.sh} > experiments/diffusion/${RUNLOG:-run36h.log} 2>&1 &
  echo "started pid $! -> $ROOT/experiments/diffusion/run36h.log"
  exit 0
fi

if tmux has-session -t "$SESS" 2>/dev/null; then
  echo "session $SESS already running:"
  tmux ls | grep "$SESS"
  echo "  attach:  tmux attach -t $SESS"
  echo "  restart: tmux kill-session -t $SESS && bash scripts/server/start_36h.sh"
  exit 0
fi

tmux new-session -d -s "$SESS" "cd $ROOT && bash scripts/server/${RUN:-run_36h.sh} 2>&1 | tee experiments/diffusion/${RUNLOG:-run36h.log}"
echo "started tmux session '$SESS'"
echo "  watch:   tmux attach -t $SESS"
echo "  detach:  Ctrl-b then d"
echo "  tail:    tail -f $ROOT/experiments/diffusion/${RUNLOG:-run36h.log}"
