#!/usr/bin/env bash
# Run on Kali: pause training, stop old tmux lifers-train, merge latest checkpoint into lifers_transformer.json, resume run.
set -eu
BR="${LIFERS_BRAIN:-/home/kali/lifers/lifers_brain}"
BR="$(cd "$BR" && pwd)"
W="$BR/weights"
mkdir -p "$W"
TS="$(date +%Y%m%dT%H%M%S)"

printf 'pause\n' >"$W/.train_control"
echo "[lifers-merge] paused -> $W/.train_control"
sleep 6

if tmux has-session -t lifers-train 2>/dev/null; then
  echo "[lifers-merge] killing tmux session lifers-train (old stack)"
  tmux kill-session -t lifers-train
else
  echo "[lifers-merge] no tmux lifers-train"
fi

sleep 2
pgrep -af train_lifers_escalate || true
# 切勿 kill pgrep 里出现的 tmux 父进程 PID，否则会干掉整个 tmux server（连 lifers-stack 一起没）。

MAIN="$W/lifers_transformer.json"
if [[ -f "$MAIN" ]]; then
  cp -a "$MAIN" "$W/lifers_transformer.json.bak.$TS"
  echo "[lifers-merge] backup main -> lifers_transformer.json.bak.$TS"
fi

CK="$W/checkpoints"
LATEST=""
if [[ -d "$CK" ]]; then
  LATEST="$(ls -t "$CK"/chunk_*.json 2>/dev/null | head -1 || true)"
fi
if [[ -n "$LATEST" && -f "$LATEST" ]]; then
  cp -a "$LATEST" "$MAIN"
  echo "[lifers-merge] merged latest checkpoint -> $MAIN"
  echo "  source=$LATEST"
  {
    echo "{\"ts\":\"$TS\",\"action\":\"merge_checkpoint_into_main\",\"from\":\"$LATEST\",\"backup\":\"lifers_transformer.json.bak.$TS\"}"
  } >>"$CK/manifest_merge.jsonl" 2>/dev/null || {
    mkdir -p "$CK"
    echo "{\"ts\":\"$TS\",\"action\":\"merge_checkpoint_into_main\",\"from\":\"$LATEST\"}" >>"$CK/manifest_merge.jsonl"
  }
else
  echo "[lifers-merge] no chunk_*.json under $CK; keeping existing $MAIN"
fi

printf 'run\n' >"$W/.train_control"
echo "[lifers-merge] control=run -> $W/.train_control"

if tmux has-session -t lifers-stack 2>/dev/null; then
  echo "[lifers-merge] lifers-stack still running — attach: tmux attach -t lifers-stack"
else
  echo "[lifers-merge] lifers-stack missing — restarting bootstrap"
  bash "$BR/scripts/remote_kali_bootstrap_train_loop.sh"
fi

pgrep -af train_lifers_escalate || echo "(no train_lifers_escalate yet — loop will spawn)"
