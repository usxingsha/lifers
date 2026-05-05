"""External run/pause/stop for long training jobs + optional checkpoint hooks."""
from __future__ import annotations

import os
from pathlib import Path


class LifersTrainingPause(Exception):
    """Control file set to pause; weights saved, exit and resume later."""

    pass


class LifersTrainingStop(Exception):
    """Control file set to stop; weights saved, exit until operator sets run again."""

    pass


def control_file_path(weights_dir: Path) -> Path:
    raw = os.environ.get("LIFERS_TRAIN_CONTROL", "").strip()
    if raw:
        return Path(raw).expanduser()
    return weights_dir / ".train_control"


def read_train_control(path: Path) -> str:
    try:
        t = path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return "run"
    if not t:
        return "run"
    return t.split()[0]


def write_default_run(path: Path) -> None:
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("run\n", encoding="utf-8")
    except OSError:
        pass


def write_train_control(path: Path, mode: str) -> None:
    """Set control file to run | pause | stop (first line only)."""
    m = (mode or "run").strip().lower().split()
    word = m[0] if m else "run"
    if word not in ("run", "pause", "stop"):
        word = "run"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(word + "\n", encoding="utf-8")
    except OSError:
        pass
