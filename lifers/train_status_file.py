"""Machine-readable training progress for tail-less monitoring (weights/.train_status.json)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def train_status_path(brain_root: Path) -> Path:
    return brain_root / "weights" / ".train_status.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Remove stale tmp from previous crash
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    blob = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    # Write + fsync to survive power loss
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(5):
        try:
            tmp.replace(path)
            break
        except OSError:
            if attempt == 4:
                raise
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.5 * (attempt + 1))


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
        # Remove heartbeat on clean exit
        _hb_path(brain_root).unlink(missing_ok=True)
    except Exception:
        pass
    clear_train_status_env()


def _hb_path(brain_root: Path) -> Path:
    return brain_root / "weights" / ".train_heartbeat.json"


def write_heartbeat(
    brain_root: Path,
    ramp_iter: int,
    ramp_max: int,
    sgd_step: int,
    sgd_total: int,
    loss: float = 0.0,
) -> None:
    """Write a lightweight heartbeat so a watchdog can detect silent death."""
    try:
        payload = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pid": os.getpid(),
            "ramp_iter": ramp_iter,
            "ramp_max": ramp_max,
            "sgd_step": sgd_step,
            "sgd_total": sgd_total,
            "loss": round(loss, 6),
        }
        _atomic_write_json(_hb_path(brain_root), payload)
    except Exception:
        pass


def recover_stale_tmp(weights_path: Path) -> bool:
    """If a .tmp weights file exists from a crash, try to recover it.

    Returns True if recovery was attempted (tmp was moved to main).
    Handles both JSON .tmp and .npz .tmp.
    """
    # Recover JSON .tmp
    json_tmp = weights_path.with_suffix(weights_path.suffix + ".tmp")
    json_recovered = False
    if json_tmp.is_file():
        try:
            tmp_size = json_tmp.stat().st_size
            main_size = weights_path.stat().st_size if weights_path.is_file() else 0
        except OSError:
            tmp_size = 0
            main_size = 0
        if tmp_size >= 1024 and (main_size == 0 or abs(tmp_size - main_size) >= 100):
            for attempt in range(5):
                try:
                    json_tmp.replace(weights_path)
                    json_recovered = True
                    break
                except OSError:
                    if attempt == 4:
                        break
                    try:
                        weights_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    time.sleep(0.5 * (attempt + 1))
        elif tmp_size > 0:
            try:
                json_tmp.unlink()
            except OSError:
                pass

    # Recover .npz .tmp
    npz_path = weights_path.with_suffix(".npz")
    npz_tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
    npz_recovered = False
    if npz_tmp.is_file():
        try:
            tmp_size = npz_tmp.stat().st_size
            main_size = npz_path.stat().st_size if npz_path.is_file() else 0
        except OSError:
            tmp_size = 0
            main_size = 0
        if tmp_size >= 1024 and (main_size == 0 or abs(tmp_size - main_size) >= 100):
            for attempt in range(5):
                try:
                    npz_tmp.replace(npz_path)
                    npz_recovered = True
                    break
                except OSError:
                    if attempt == 4:
                        break
                    try:
                        npz_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    time.sleep(0.5 * (attempt + 1))
        elif tmp_size > 0:
            try:
                npz_tmp.unlink()
            except OSError:
                pass

    return json_recovered or npz_recovered


def detect_crash(brain_root: Path) -> dict | None:
    """Check heartbeat to see if previous run died mid-training.

    Returns heartbeat dict if crash detected (heartbeat exists but no clean status),
    None otherwise.
    """
    hb = _hb_path(brain_root)
    if not hb.is_file():
        return None
    try:
        return json.loads(hb.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
