#!/usr/bin/env python3
"""One embodied tick (physics + eyes + decision). Respects weights/.train_control pause/stop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or lifers_brain/
_ROOT = Path(__file__).resolve().parent.parent
if (_ROOT / "lifers_brain").is_dir():
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_ROOT, help="lifers_brain root")
    args = ap.parse_args()
    root: Path = args.root.resolve()
    from lifers_brain.embodied import run_embodied_tick

    out = run_embodied_tick(root)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
