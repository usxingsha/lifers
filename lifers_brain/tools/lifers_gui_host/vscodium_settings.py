"""
从仓库根 tools/vscodium_editor_defaults.json 读取与 VSCodium 工作区一致的编辑器默认值，
供自研 GUI 映射字体、留白、暗色主题等（不嵌入 VSCodium 二进制，仅对齐配置源）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def repo_root_from_brain(brain_root: Path) -> Path:
    return (brain_root.resolve()).parent


def load_vscodium_defaults_json(repo_root: Path) -> Dict[str, Any]:
    p = repo_root / "tools" / "vscodium_editor_defaults.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def gui_theme_from_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """抽取 UI 可用子集（CSS 变量 / 布局）。"""
    bg = "#1e1e1e"
    fg = "#d4d4d4"
    if (raw.get("workbench.colorTheme") or "").lower().find("light") >= 0:
        bg = "#ffffff"
        fg = "#1e1e1e"
    fs = raw.get("editor.fontSize", 14)
    ff = raw.get("editor.fontFamily", "Consolas, monospace")
    tab = raw.get("editor.tabSize", 4)

    def _intish(x: Any, default: int) -> int:
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip())
        return default

    return {
        "surface": bg,
        "text": fg,
        "fontSizePx": _intish(fs, 14),
        "fontFamily": str(ff),
        "tabSize": _intish(tab, 4),
        "titleBarStyle": raw.get("window.titleBarStyle", "custom"),
        "sourceFile": "tools/vscodium_editor_defaults.json",
    }
