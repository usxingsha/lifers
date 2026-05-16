#!/usr/bin/env bash
set -euo pipefail
BRAIN="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BRAIN"
export PYTHONPATH="."
exec python3 scripts/lifers_run_all_checks.py
