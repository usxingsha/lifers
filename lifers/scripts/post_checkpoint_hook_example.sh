#!/usr/bin/env bash
# Example for LIFERS_POST_CHECKPOINT_CMD — copy/sync latest shard then let trainer pause (optional).
# Env set by train_lifers_escalate: LIFERS_ROOT LIFERS_CHECKPOINT_JSON LIFERS_CHECKPOINT_B LIFERS_CUMULATIVE_EST
set -euo pipefail
ROOT="${LIFERS_ROOT:-}"
JSON="${LIFERS_CHECKPOINT_JSON:-}"
B="${LIFERS_CHECKPOINT_B:-?}"
if [[ -z "$ROOT" || -z "$JSON" ]]; then
  echo "[post-cp] missing LIFERS_ROOT or LIFERS_CHECKPOINT_JSON" >&2
  exit 0
fi
echo "[post-cp] B_floor=$B checkpoint=$JSON cumulative_est=${LIFERS_CUMULATIVE_EST:-}"
# rsync/scp to your workstation, tarball, object store, etc.:
# rsync -av "$ROOT/weights/" user@host:~/lifers_inbox/weights/
exit 0
