#!/usr/bin/env python3
"""入口：自研 GUI + Bridge 宿主（见 tools/lifers_gui_host/host.py）。"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tools.lifers_gui_host.host import main as host_main

    return int(host_main())


if __name__ == "__main__":
    raise SystemExit(main())
