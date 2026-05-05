"""Machine-readable training progress for tail-less monitoring (weights/.train_status.json)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def train_status_path(brain_root: Path) -> Path:
    return brain_root / "weights" / ".train_status.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    blob = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp.write_text(blob, encoding="utf-8")
    tmp.replace(path)


def publish_escalate_snapshot(
    brain_root: Path,
    *,
    phase: str,
    ramp_iter: int | None = None,
    ramp_max: int | None = None,
    tier_est_m: float | None = None,
    max_vocab: int | None = None,
    d_model: int | None = None,
    d_ff: int | None = None,
    max_seq: int | None = None,
    steps: int | None = None,
    sgd_step: int | None = None,
    sgd_total: int | None = None,
    weight_rel: str = "weights/lifers_transformer.json",
    message: str | None = None,
    cumulative_est_g: float | None = None,
) -> None:
    """Write full snapshot (replace each time)."""
    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(),
        "phase": phase,
        "brain_root": str(brain_root.resolve()),
        "weight_file": weight_rel,
    }
    if message:
        payload["message"] = message
    if cumulative_est_g is not None:
        payload["cumulative_est_g"] = round(cumulative_est_g, 6)
    rmax = ramp_max if ramp_max is not None else 1
    rcur = ramp_iter if ramp_iter is not None else 1
    if ramp_iter is not None and ramp_max is not None:
        payload["ramp"] = {
            "iter": ramp_iter,
            "max": ramp_max,
            "pct": round(100.0 * ramp_iter / max(rmax, 1), 2),
        }
    if tier_est_m is not None:
        payload["tier_est_params_m"] = round(tier_est_m, 6)
    arch = {}
    if max_vocab is not None:
        arch["max_vocab"] = max_vocab
    if d_model is not None:
        arch["d_model"] = d_model
    if d_ff is not None:
        arch["d_ff"] = d_ff
    if max_seq is not None:
        arch["max_seq"] = max_seq
    if steps is not None:
        arch["steps_this_tier"] = steps
    if arch:
        payload["architecture"] = arch

    if sgd_step is not None and sgd_total is not None:
        st = max(int(sgd_total), 1)
        sc = max(0, min(int(sgd_step), st))
        sgd_pct = 100.0 * sc / st
        payload["sgd"] = {
            "step": sc,
            "total_steps": st,
            "pct": round(sgd_pct, 2),
            "vocab_size": max_vocab,
            "d_model": d_model,
        }
        # Outer ramp × inner SGD rough overall progress (for humans).
        if ramp_iter is not None and ramp_max is not None:
            portion_tier = sc / st
            overall = 100.0 * ((rcur - 1) + portion_tier) / max(rmax, 1)
            payload["overall_pct_approx"] = round(min(100.0, max(0.0, overall)), 2)

    _atomic_write_json(train_status_path(brain_root), payload)


def refresh_sgd_status(cur_step: int, total_steps: int, V: int, D: int) -> None:
    """Called from train_sgd inner loop when LIFERS_TRAIN_STATUS_BRAIN_ROOT is set."""
    raw = os.environ.get("LIFERS_TRAIN_STATUS_BRAIN_ROOT", "").strip()
    if not raw:
        return
    brain_root = Path(raw)
    try:
        ri = int(os.environ.get("LIFERS_TRAIN_STATUS_RAMP_ITER", "1"))
        rm = int(os.environ.get("LIFERS_TRAIN_STATUS_RAMP_MAX", "1"))
        est_m = float(os.environ.get("LIFERS_TRAIN_STATUS_TIER_EST_M", "0"))
        mv = int(os.environ.get("LIFERS_TRAIN_STATUS_MAX_V", str(V)))
        dm = int(os.environ.get("LIFERS_TRAIN_STATUS_D", str(D)))
        df = int(os.environ.get("LIFERS_TRAIN_STATUS_DFF", "0"))
        ms = int(os.environ.get("LIFERS_TRAIN_STATUS_MS", "0"))
        st_env = os.environ.get("LIFERS_TRAIN_STATUS_STEPS", "").strip()
        steps_hint = int(st_env) if st_env.isdigit() else total_steps
    except (TypeError, ValueError):
        return
    publish_escalate_snapshot(
        brain_root,
        phase="sgd",
        ramp_iter=ri,
        ramp_max=rm,
        tier_est_m=est_m,
        max_vocab=mv,
        d_model=dm,
        d_ff=df if df > 0 else None,
        max_seq=ms if ms > 0 else None,
        steps=steps_hint,
        sgd_step=cur_step,
        sgd_total=total_steps,
        message=f"train_sgd V={V} D={D} step {cur_step}/{total_steps}",
    )


def clear_train_status_env() -> None:
    for k in list(os.environ.keys()):
        if k.startswith("LIFERS_TRAIN_STATUS_"):
            del os.environ[k]


def finalize_train_status(brain_root: Path, phase: str, message: str = "") -> None:
    """Last write + drop env keys (process exit)."""
    try:
        publish_escalate_snapshot(brain_root, phase=phase, message=message or phase)
    except Exception:
        pass
    clear_train_status_env()
