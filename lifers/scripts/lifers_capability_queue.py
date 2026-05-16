#!/usr/bin/env python3
"""Rotate LIFERS_TRAIN_SUITE_DIR across capability tracks (no fixed product 'gear ceiling')."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_queue() -> dict:
    p = _root() / "config" / "capability_queue.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _active_path() -> Path:
    return _root() / "weights" / ".capability_queue_active.json"


def _load_active() -> dict:
    p = _active_path()
    if not p.is_file():
        return {"track_index": 0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"track_index": 0}


def _save_active(o: dict) -> None:
    p = _active_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _track(queue: dict, idx: int) -> dict:
    tracks = queue.get("tracks") or []
    if not tracks:
        raise SystemExit("capability_queue.json: missing tracks")
    return tracks[idx % len(tracks)]


def cmd_show() -> int:
    queue = _load_queue()
    act = _load_active()
    idx = int(act.get("track_index", 0))
    t = _track(queue, idx)
    suite = (_root() / str(t.get("suite_dir", ""))).resolve()
    print(f"index={idx} id={t.get('id')} suite_dir={suite}")
    return 0


def cmd_env(*, shell: str) -> int:
    queue = _load_queue()
    act = _load_active()
    idx = int(act.get("track_index", 0))
    t = _track(queue, idx)
    rel = str(t.get("suite_dir", "eval/suites/v001")).strip() or "eval/suites/v001"
    suite = (_root() / rel).resolve()
    if shell == "fish":
        print(f"set -gx LIFERS_TRAIN_SUITE_DIR {suite}")
    elif shell == "pwsh":
        print(f'$env:LIFERS_TRAIN_SUITE_DIR = "{suite}"')
    else:
        print(f"export LIFERS_TRAIN_SUITE_DIR={suite}")
    return 0


def cmd_advance() -> int:
    queue = _load_queue()
    act = _load_active()
    n = len(queue.get("tracks") or [])
    if n < 1:
        return 1
    idx = (int(act.get("track_index", 0)) + 1) % n
    act["track_index"] = idx
    _save_active(act)
    t = _track(queue, idx)
    suite = (_root() / str(t.get("suite_dir", ""))).resolve()
    print(f"advanced -> index={idx} id={t.get('id')} suite_dir={suite}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("show", help="Print active track and resolved suite path")
    p_env = sp.add_parser("env", help="Print shell export for LIFERS_TRAIN_SUITE_DIR")
    p_env.add_argument(
        "--shell",
        choices=("sh", "fish", "pwsh"),
        default="sh",
        help="Output syntax (default: POSIX export)",
    )
    sp.add_parser("advance", help="Move to next track (wrap) and print new suite")
    args = ap.parse_args()
    if args.cmd == "show":
        return cmd_show()
    if args.cmd == "env":
        return cmd_env(shell=args.shell)
    if args.cmd == "advance":
        return cmd_advance()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
