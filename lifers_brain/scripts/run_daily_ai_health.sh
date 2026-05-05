#!/usr/bin/env bash
# Daily AI health (logic/coherence, full_system_check, bridge latency, context throughput).
# Reports: lifers_brain/state/daily_health/
#
# Optional auto-remediation (local workspace_custom.json):
#   LIFERS_DAILY_HEALTH_AUTO_REMEDIATE=1 ./scripts/run_daily_ai_health.sh
#
# Cron (Kali), e.g. 04:30 daily:
#   30 4 * * * cd /home/kali/lifers/lifers_brain && /usr/bin/env bash scripts/run_daily_ai_health.sh >>/tmp/lifers_daily_health.log 2>&1

set -euo pipefail
BRAIN="$(cd "$(dirname "$0")/.." && pwd)"
export LIFERS_ROOT="${LIFERS_ROOT:-$BRAIN}"
PY="${BRAIN}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi
exec "$PY" "$BRAIN/eval/daily_ai_health.py"
