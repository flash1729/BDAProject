#!/usr/bin/env bash
# Full run in the background-friendly way: keep the Mac awake (caffeinate -i) and
# yield CPU to interactive apps (nice -n 10). Extra args go to train.py, e.g.
#   ./run.sh                 # 5 x 5-fold, all methods
#   ./run.sh --resume        # continue after a kill / reboot
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
mkdir -p results
# -u: unbuffered so the log updates live; tee -a so a --resume appends instead of wiping the log
caffeinate -i nice -n 10 "$PY" -u train.py "$@" 2>&1 | tee -a results/train.log
