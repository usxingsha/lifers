"""Central knobs for latency / throughput (training, bridge, HTTP)."""
from __future__ import annotations

import os


def max_speed_enabled() -> bool:
    v = os.environ.get("LIFERS_MAX_SPEED", "").strip().lower()
    return v in ("1", "true", "yes", "max", "on")


def pause_poll_seconds() -> float:
    raw = os.environ.get("LIFERS_PAUSE_POLL_SEC", "").strip()
    if raw:
        return max(0.02, float(raw))
    return 0.15 if max_speed_enabled() else 2.0


def train_save_every(steps: int) -> int:
    """Write weights JSON every N steps (large models: N>1 saves massive I/O)."""
    raw = os.environ.get("LIFERS_TRAIN_SAVE_EVERY", "").strip()
    if raw:
        return max(1, int(raw))
    if max_speed_enabled():
        # default: ~5% of steps, clamped
        return max(1, min(100, max(1, steps // 20)))
    return 1


def train_control_every_steps(log_every: int) -> int:
    raw = os.environ.get("LIFERS_TRAIN_CONTROL_EVERY", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, log_every * (5 if max_speed_enabled() else 1))


def http_timeout_seconds(base: float) -> float:
    cap = os.environ.get("LIFERS_HTTP_TIMEOUT_CAP", "").strip()
    if cap:
        return max(0.5, min(base, float(cap)))
    if max_speed_enabled():
        return max(0.5, min(base, 12.0))
    return base


def local_lm_max_chars(default_mc: int) -> int:
    raw = os.environ.get("LIFERS_LOCAL_LM_MAX_CHARS", "").strip()
    if raw:
        return max(32, int(raw))
    if max_speed_enabled():
        return min(default_mc, max(64, int(os.environ.get("LIFERS_SPEED_MAX_CHARS", "140"))))
    return default_mc
