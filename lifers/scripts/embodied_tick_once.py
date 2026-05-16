#!/usr/bin/env python3
"""
One embodied tick (physics + eyes + decision). Respects weights/.train_control pause/stop.

与 Agents Chat / Bridge 并行：不读写会话 JSON；动态 NPC 多体扩展见 stack.embodied_world.dynamic_npc
与 lifers.embodied.coordinator（当前单 PhysBody）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or lifers/
_ROOT = Path(__file__).resolve().parent.parent
if (_ROOT / "lifers").is_dir():
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_ROOT, help="lifers root")
    args = ap.parse_args()
    root: Path = args.root.resolve()
    from lifers.embodied import run_embodied_tick

    out = run_embodied_tick(root)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
