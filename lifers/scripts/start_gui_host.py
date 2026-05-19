"""独立GUI Host启动器 — 绕过tools.py/tools目录命名冲突"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "tools" / "lifers_gui_host" / "host.py"
spec = importlib.util.spec_from_file_location("lifers_gui_host", str(MODULE_PATH))
host = importlib.util.module_from_spec(spec)
sys.modules["lifers_gui_host"] = host
spec.loader.exec_module(host)
raise SystemExit(host.main())
