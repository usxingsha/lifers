# Example environment overrides for kali_train_escalate_loop.sh.
# Copy to scripts/kali_escalate_env.sh and edit as needed.
#
# For low-RAM Kali hosts (< 8 GiB):
#   LIFERS_ESCALATE_MAX_TIER=12       # stop before OOM (~70M params)
#   LIFERS_ESCALATE_PAUSE_ON_MEMERROR=1
#
# For mid-range Kali (8-16 GiB):
#   LIFERS_ESCALATE_MAX_TIER=18       # ~400M params
#
# For persistent training loops:
#   LIFERS_ESCALATE_UNLIMITED=1
#   LIFERS_RAMP_MAX_ITERS=999999

# ---- Backprop training settings ----
# These are passed through to train_backprop_minimal().
# Increase steps for better quality; d_model=128 needs ~20K steps.
export TT_STEPS="${TT_STEPS:-10000}"
export TT_DMODEL="${TT_DMODEL:-96}"
export TT_DFF="${TT_DFF:-256}"
export TT_VOCAB="${TT_VOCAB:-200}"
export TT_MAXSEQ="${TT_MAXSEQ:-56}"
export TT_LR="${TT_LR:-3e-4}"

# ---- OpenBLAS threading (Kali) ----
# Set to match physical cores for best training throughput.
if [ -z "${OMP_NUM_THREADS:-}" ] && [ -z "${OPENBLAS_NUM_THREADS:-}" ]; then
  if command -v nproc >/dev/null 2>&1; then
    export OMP_NUM_THREADS="$(nproc)"
  fi
fi

# ---- Ramp control (low-RAM safety) ----
# export LIFERS_ESCALATE_MAX_TIER=16
# export LIFERS_ESCALATE_PAUSE_ON_MEMERROR=1

# ---- Corpus ----
# Point to a larger corpus directory if available:
# export LIFERS_TRAIN_SUITE_DIR=/path/to/larger/corpus
