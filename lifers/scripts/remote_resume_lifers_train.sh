#!/usr/bin/env bash
set -euo pipefail
BR="${LIFERS_BRAIN:-$HOME/lifers/lifers}"
mkdir -p "$BR/weights"
echo run >"$BR/weights/.train_control"
echo "OK_run $BR/weights/.train_control"
