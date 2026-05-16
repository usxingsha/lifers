"""Terminal / log-friendly progress bars for long-running trainers."""
from __future__ import annotations

import os
import sys
from typing import TextIO


def _bar_chars() -> tuple[str, str]:
    v = os.environ.get("LIFERS_PROGRESS_ASCII", "").strip().lower()
    if v in ("1", "true", "yes"):
        return "#", "-"
    if v in ("0", "false", "no"):
        return "█", "░"
    # Windows default consoles are often GBK/CP936 — block chars raise UnicodeEncodeError.
    if sys.platform == "win32":
        return "#", "-"
    return "█", "░"


def format_bar(cur: int, total: int, width: int = 24) -> tuple[str, float]:
    total_i = max(int(total), 1)
    cur_i = max(0, min(int(cur), total_i))
    frac = cur_i / total_i
    filled, w = int(round(width * frac)), max(1, int(width))
    if filled > w:
        filled = w
    fch, ech = _bar_chars()
    return f"{fch * filled}{ech * (w - filled)}", frac


def write_progress_line(
    stream: TextIO,
    cur: int,
    total: int,
    *,
    prefix: str = "",
    width: int = 22,
    tty: bool | None = None,
    newline_when_not_tty: bool = True,
) -> None:
    """TTY: single-line \\r update. Non-TTY: one line per call (for tee / nohup logs)."""
    bar, frac = format_bar(cur, total, width)
    pct = 100.0 * frac
    text = f"{prefix}[{bar}] {cur}/{total} {pct:.1f}%"
    is_tty = stream.isatty() if tty is None else tty
    if is_tty:
        stream.write(f"\r{text}\033[K")
    elif newline_when_not_tty:
        stream.write(text + "\n")
    else:
        stream.write(text)
    stream.flush()


def end_progress_line(stream: TextIO, *, tty: bool | None = None) -> None:
    """After final \\r bar, start a fresh line on TTY."""
    is_tty = stream.isatty() if tty is None else tty
    if is_tty:
        stream.write("\n")
        stream.flush()


def default_progress_stream() -> TextIO:
    return sys.stderr
