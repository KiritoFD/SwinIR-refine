#!/usr/bin/env bash
# Training-free deterministic upper-bound probe (oracle.py): E_align / E_null /
# E_noise(MAD) + composite ceiling, on the official Test protocol.  No model, no
# leakage -- the correct instrument for a dataset ceiling.  (The earlier
# overfit-to-Test idea was scrapped: a big net memorises GT noise/misalignment, so
# it measures model capacity, not the data's information limit, and can't bound PSNR.)
set -uo pipefail
cd /home/ds/realsr || exit 1
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
for sc in 2 3 4; do
  echo "########## ORACLE x${sc} ##########"
  "$PY" -u -m diffusion.oracle --scale "$sc"
done
echo "ORACLE DONE"
