#!/usr/bin/env bash
# Idempotent launcher for chain_after2.sh.
#
# Twice now a restart has left TWO chain instances alive, because `pkill -f
# chain_after2` matches the ssh wrapper's own command line and kills the shell
# before it can report (or kill) anything.  Two instances are not harmless: the
# second one does `tmux kill-session -t realsrpt` and would destroy the
# pretraining run the first one just started.
#
# An atomic mkdir is the simplest lock that needs no extra tooling.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
LOCK=/tmp/chain_after2.lock
cd "$ROOT" || exit 1

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "chain_after2 already running (lock $LOCK held):"
  ps -eo pid,lstart,args | grep "[c]hain_after2.sh" | grep -v "bash -c" || true
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

exec bash scripts/server/chain_after2.sh
