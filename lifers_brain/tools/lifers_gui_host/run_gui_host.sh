#!/usr/bin/env bash
set -euo pipefail
BRAIN="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BRAIN"
export PYTHONPATH="."
exec python3 scripts/run_lifers_gui_host.py "$@"
